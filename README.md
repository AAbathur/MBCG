# Multi-modal image-text content Based Comment Generation
code for paper "Stimulating Conversation-style Emergencies of Multi-Modal LMs"

## Installation
Our work is based on the released source code of [OFA](https://github.com/OFA-Sys/OFA) and [BLIP2](https://github.com/salesforce/LAVIS), 
there are only a few changes on their source code to apply the proposed MBCG task.

### OFA
Install the source code of OFA
```bash
git clone https://github.com/OFA-Sys/OFA
cd OFA
pip install -r requirements.txt
```
The changes include:
1. The ./tasks/pretrain_tasks/unify_task.py file should be replaced with [this file](OFA/unify_task.py) to build the image-text-comments triplet dataset.
2. The ./data/pretrain_data/unify_dataset.py file should be replaced with [this file](OFA/unify_dataset.py) to process the image-text-comments triplet data samples.
3. Adding the MBCG [pre-training script](OFA/pretrain_ofa_base_with_mbcg.sh) to the directory ./run_scripts/pretraining

### BLIP2
Install the source code of BLIP2
```bash
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e 
```
The changes include:
1. The ./lavis/models/blip2_models/blip2_qformer.py file should be replaced with [this file](BLIP2/blip2_qformer.py) to apply MBCG task for BLIP2 stage 1 pre-training.
2. Adding image-text-comments processing [dataset file](BLIP2/cmt_gen_dataset.py) to ./lavis/datasets/datasets/cmt_ins_datasets.py.
3. Adding the dataset [config file](BLIP2/caption_cmt_mix.yaml) to ./lavis/configs/datasets/cmt_mix/caption_cmt_mix.yaml.
4. Adding the new [builder file](BLIP2/cmt_builder.py) to ./lavis/datasets/builders/cmt_builder.py to register the new dataset.
5. Adding [this](BLIP2/pretrain_stage1_with_mbcg.yaml) yaml file to the directory of ./lavis/projects/blip2/train to set the configure for stage 1 pre-training with the MBCG task
6. Adding [this](BLIP2/pretrain_stage2_with_mbcg.yaml) yaml file to the directory of ./lavis/projects/blip2/train to set the configure for stage 2 pre-training with the MBCG task

## Data Preparation

### Pre-training data for OFA
Following the dataset processing methods of OFA, we process the image-text-comments triplet dataset into .tsv file, which has the same format as the [vision_langauge_exmamples.tsv file](https://ofa-beijing.oss-cn-beijing.aliyuncs.com/datasets/pretrain_data/pretrain_data_examples.zip).

ins_cmt_en.tsv: Each line contains uniq-id, image (base64 string), "", post, cmts(concatenated by "&&"), "", "ins_cmt"(dataset name) and "cmt" (task type).

### Pre-training data for BLIP2
The MBCG-EN data is processed into .json file. 
Each sample is formatted as {"pid":pid, "post":post, "cmts": [cmt1, cmt2, cmt3, ...]}

The MBCG-EN is mixed with three image caption datasets (COCO caption, Visual Genome Captions, SBU Captions).
Each image caption sample is formatted as {"image_id":image_id, "caption: caption}.

## Pre-training with the MBCG task

## Pre-training for OFA
After preparing the pre-training data and employing the changes, then can run the below script to pre-training OFA-base model with the MBCG task
```bash
cd OFA
bash run_scripts/pretraining/pretrain_ofa_base_with_mbcg.sh
```

## Pre-training for BLIP2
After prepraing the pre-training data and employing, then can run the blew script to pre-training BLIP2 model with the MBCG task
```bash
## stage 1 pre-training
python -m torch.distributed.run --nproc_per_node=<gpu_num> train.py --cfg-path lavis/projects/blip2/train/pretrain_stage1_with_mbcg.yaml
```
```bash
## stage 2 pre-training
python -m torch.distributed.run --nproc_per_node=<gpu_num> train.py --cfg-path lavis/projects/blip2/train/pretrain_stage2_with_mbcg.yaml
```


## Results on the Conversation-style Tasks
The evaluation experiments are conducted on four publicly released conversation-style tasks, including Visual Dialog, Image Grounded Conversation (IGC), MMDialog, and Image-Chat. The results are shown as below, "*" means the model is fine-tuned on the evaluation dataset.

<table border="1", width="100%"> 
    <tr align='center'>
    <td> Models </td>
    <td colspan="6"> Visual Diag </td>
    <td colspan="2"> IGC </td>
    <td colspan="3"> MMDialog </td>
    <td colspan="6"> Image-Chat </td>
    <tr>
    <tr align='center'>
    <td> </td>
    <td colspan="3"> val </td>
    <td colspan="3"> test </td>
    <td colspan="2"> test </td>
    <td colspan="3"> test </td>
    <td colspan="3"> val </td>
    <td colspan="3"> test </td>
    <tr align='center'>
    <td> </td> <td> R@1 </td> <td> R@5 </td> <td> R@10 </td> <td> R@1 </td> <td> R@5 </td> <td> R@10 </td> <td> R@1 </td> <td> R@3 </td> <td> R@1 </td> <td> R@5 </td> <td> R@10 </td> <td> R@1 </td> <td> R@5 </td> <td> R@10 </td> <td> R@1 </td> <td> R@5 </td> <td> R@10 </td>
    <tr align='center'>
    <td> Two-step* </td> <td> 59.2</td> <td> 84.6</td> <td> 90.8</td> <td> 58.2</td> <td> 83.9</td> <td> 90.8</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td>
    <tr align='center'>
    <td> 5xFGA+LS* </td> <td> 59.2</td> <td> 88.6</td> <td> 94.5</td> <td> 58.3</td> <td> 87.6</td> <td> 94.5</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td>
    <tr align='center'>
    <td> DE++* </td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> 18.2</td> <td> 27.0</td> <td> 31.7</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td>
    <tr align='center'>
    <td> TransResNet<sub>ret</sub>* </td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> -</td> <td> 37.6</td> <td> -</td> <td> -</td>
    <tr align='center'>
    <td> OFA </td> <td> 1.9</td> <td> 7.3</td> <td> 13.0</td> <td> 1.5</td> <td> 7.6</td> <td> 12.9</td> <td> 29.0</td> <td> 72.9</td> <td> 1.9</td> <td> 8.4</td> <td> 26.3</td> <td> 16.3</td> <td> 49.1</td> <td> 61.4</td> <td> 15.9</td> <td> 49.3</td> <td> 61.5</td>
    <tr align='center'>
    <td> OFA+MBCG </td> <td> 4.9</td> <td> 18.1</td> <td> 31.9</td> <td> 3.8</td> <td> 15.8</td> <td> 29.1</td> <td> 40.1</td> <td> 81.3</td> <td> 13.6</td> <td> 25.2</td> <td> 32.1</td> <td> 30.4</td> <td> 53.2</td> <td> 63.7</td> <td> 29.5</td> <td> 52.7</td> <td> 63.5</td>
    <tr align='center'>
    <td> BLIP2 </td> <td> 10.8</td> <td> 23.2</td> <td> 32.5</td> <td> 10.9</td> <td> 22.6</td> <td> 31.3</td> <td> 46.4</td> <td> 78.1</td> <td> 21.2</td> <td> 35.5</td> <td> 41.6</td> <td> 29.4</td> <td> 48.7</td> <td> 58.3</td> <td> 28.8</td> <td> 48.6</td> <td> 58.0</td>
    <tr align='center'>
    <td> BLIP2+MBCG </td> <td> 12.2</td> <td> 25.6</td> <td> 35.8</td> <td> 12.5</td> <td> 25.0</td> <td> 35.5</td> <td> 59.6</td> <td> 89.4</td> <td> 27.3</td> <td> 45.9</td> <td> 53.8</td> <td> 34.7</td> <td> 59.3</td> <td> 68.7</td> <td> 34.3</td> <td> 58.5</td> <td> 68.9</td>
    <tr>


</table>




## License
This project is released under the [MIT License](LICENSE).
