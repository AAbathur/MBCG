#!/usr/bin/env

export MASTER_PORT=1055
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=8


bpe_dir=./utils/BPE
user_dir=./ofa_module

restore_file=./ofa_official_models/ofa_base.pt

data_dir=./the/processed/cmt/file/path
neg_sample_dir=./dataset/negative_sample
data=${data_dir}

## following the OFA paper to prepare these four types of datas
image_text_data=./image-text/pair/data/path ## following the OFA paper
text_data=./pure/text/data/path
image_data=./pure/image/data/path
detection_data=./object/detection/data/path

save_path=./checkpoint/save/path


image_text_selected_cols=0,1,2,3,4,5,6,7
selected_cols=0,1,2,3,4,5,6,7
text_selected_cols=0,1
image_selected_cols=0,3
detection_selected_cols=0,1,2

log_interval=30
task=unify_task
arch=ofa_base
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.0
lr=5e-5


max_epoch=10
warmup_ratio=0.01
batch_size=32
update_freq=1
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=50
max_tgt_length=30
num_bins=1000
patch_image_size=384
sample_patch_num=196
max_image_size=512



python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ./train.py \
  $data \
  --selected-cols=${selected_cols} \
  --bpe-dir=${bpe_dir} \
  --user-dir=${user_dir} \
  --image-text-data=${image_text_data} \
  --image-text-selected-cols=${image_text_selected_cols} \
  --text-data=${text_data} \
  --text-selected-cols=${text_selected_cols} \
  --image-data=${image_data} \
  --image-selected-cols=${image_selected_cols} \
  --detection-data=${detection_data} \
  --detection-selected-cols=${detection_selected_cols} \
  --restore-file=${restore_file} \
  --reset-optimizer --reset-dataloader --reset-meters \
  --save-dir=${save_path} \
  --neg-sample-dir=${neg_sample_dir} \
  --task=${task} \
  --arch=${arch} \
  --criterion=${criterion} \
  --label-smoothing=${label_smoothing} \
  --batch-size=${batch_size} \
  --update-freq=${update_freq} \
  --encoder-normalize-before \
  --decoder-normalize-before \
  --share-decoder-input-output-embed \
  --share-all-embeddings \
  --layernorm-embedding \
  --patch-layernorm-embedding \
  --code-layernorm-embedding \
  --resnet-drop-path-rate=${resnet_drop_path_rate} \
  --encoder-drop-path-rate=${encoder_drop_path_rate} \
  --decoder-drop-path-rate=${decoder_drop_path_rate} \
  --dropout=${dropout} \
  --attention-dropout=${attention_dropout} \
  --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=5.0 \
  --lr-scheduler=polynomial_decay --lr=${lr} \
  --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
  --log-format=simple --log-interval=${log_interval} \
  --fixed-validation-seed=7 \
  --keep-last-epochs=50 \
  --save-interval=10 \
  --save-interval-updates=10000 \
  --disable-validation \
  --max-src-length=${max_src_length} \
  --max-tgt-length=${max_tgt_length} \
  --add-type-embedding \
  --scale-attn \
  --scale-fc \
  --scale-heads \
  --disable-entangle \
  --num-bins=${num_bins} \
  --patch-image-size=${patch_image_size} \
  --sample-patch-num=${sample_patch_num} \
  --max-image-size=${max_image_size} \
  --fp16 \
  --fp16-scale-window=128 \
  --find-unused-parameters \
  --num-workers=0 \
  --end-learning-rate=1e-7 \
