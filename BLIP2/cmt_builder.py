

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry

from lavis.datasets.datasets.cmt_ins_datasets import CMTINSDataset


@registry.register_builder("cmt_ins")
class CmtInsBuilder(BaseDatasetBuilder):
    train_dataset_cls = CMTINSDataset
    eval_dataset_cls = CMTINSDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cmt_ins/default.yaml",
    }