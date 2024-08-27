import torch
import lmdb
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO
import os
import random

from lavis.datasets.datasets.base_dataset import BaseDataset

class CMTINSDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        #super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.key_file = vis_root[0]
        self.img_dir = vis_root
        self.key_file = os.path.join(os.path.dirname(vis_root), "key_file.txt")
        self.key2idx, self.idx2key = self.get_img_keys(self.key_file)
        keys = set(self.key2idx.keys())

        self.all_text = []
        for tpath in ann_paths:
            self.all_text += self.get_text_info(tpath, keys)

        self.img_env = lmdb.open(self.img_dir, readonly=True, lock=False, readahead=False, meminit=False)

    def get_img_keys(self, keyfile):
        key2idx, idx2key = {}, {}
        with open(keyfile, "r", encoding="utf-8") as f1:
            for idx, line in enumerate(f1):
                key2idx[line.strip()] = idx
                idx2key[idx] = line.strip()
        return key2idx, idx2key
    
    def get_raw_img(self, pid):
        with self.img_env.begin(write=False) as txn:
            value = txn.get(pid.encode())
        image = Image.open(BytesIO(value))
        image = image.convert('RGB')
        image = self.vis_processor(image)
        return image

    def get_text_info(self, tpath, keys):
        all_line = []
        with open(tpath, "r", encoding="utf-8") as f1:
            for i, line in enumerate(f1):
                pid, post, cmt_str = line.strip().split(" #EOS# ")
                cmts = cmt_str.split(" #EOC# ")
                if pid not in keys:
                    continue
                all_line.append([pid, post, cmts])
        return all_line
    
    def __len__(self, ):
        return len(self.all_text)
    
    def __getitem__(self, index):
        pid, post, cmts = self.all_text[index]

        img = self.get_raw_img(pid)
        
        post = self.text_processor(post)
        if len(cmts) >= 3:
            cmts = random.sample(cmts, 3)
        cmts = [self.text_processor(cmt) for cmt in cmts]

        return {
            "did": pid,
            "post": post,
            "img": img,
            "cmts": cmts
        }

    def collater(self, samples):
        did_list, post_list, image_list, cmt_list = [], [], [], []

        for sample in samples:
            did_list.append(sample['did'])
            image_list.append(sample['img'])
            post_list.append(sample['post'])
            
            cmts = sample['cmts']
            cmt_list.append(cmts)
        
        return {
            "image": torch.stack(image_list, dim=0),
            "post_input": post_list,
            "comment": cmt_list,
        }




