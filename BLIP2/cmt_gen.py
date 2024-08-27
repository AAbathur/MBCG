"""
参考lavis/tasks/vqa.py中的方式，自定义评论生成任务的task
"""

import logging
import json
import os

import lavis.common.dist_utils as dist_utils
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.common.dist_utils import main_process

@registry.register_task("cmt_gen")
class CmtGenTask(BaseTask):
    def __init__(
            self,
            num_beams,
            max_cmt_len,
            min_cmt_len,
            report_metric
    ):
        super().__init__()
        self.num_beams = num_beams
        self.max_cmt_len = max_cmt_len
        self.min_cmt_len = min_cmt_len

        self.report_metic = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_cmt_len = run_cfg.max_cmt_len
        min_cmt_len = run_cfg.min_cmt_len

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_cmt_len=max_cmt_len,
            min_cmt_len=min_cmt_len,
            report_metric=report_metric,
        )

    def train_step(self, model, samples):
        output = model.forward(samples)
        loss_dict = {}
        for k,v in output.items():
            if "loss" in k:
                loss_dict[k] = v
        return output["loss"], loss_dict

    def valid_step(self, model, samples):
        results = []

        cmts = model.generate_cmt(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_cmt_len,
            min_length=self.min_cmt_len,
        )

        dids = samples['did']
        for cmt, did in zip(cmts, dids):
            results.append({"did": did, "cmt": cmt})

        return results
    
    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="did",
        )

        if self.report_metic:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}
        
        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        cmt_gt_root = os.path.join(registry.get_path("cache_root"), "cmt_gt")
        cmt_val = coco_caption_eval(cmt_gt_root, eval_result_file, split_name)

        agg_metrics = cmt_val.eval["CIDEr"] + cmt_val.eval["Bleu-4"]
        log_stats = {split_name: {k: v for k,v in cmt_val.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"),  "evaluate.txt"), "a",
        ) as f:
            f.write(json.dumps(log_stats) + "\n")
        
        cmt_res = {k: v for k, v in cmt_val.eval.items()}
        cmt_res["agg_metrics"] = agg_metrics

        return cmt_res





# TODO better structure for this.
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_url


def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {
        "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
        "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
    }
    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval
