#! /bin/bash
MODEL_PATH="PATH_TO_MODEL_DIR"
python src/prefix_heat_map.py \
    --T5_checkpoint ../save/models/small/ \
    --gpu_id 0 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --GPU 1 \
    --except_domain train \
    --slot_lang question \
    --worker_number 8 \
    --warm_up_steps 1000 \
    --seed 557 \
    --saving_dir save \
    --down_proj 128 \
    --prefix_length 10 \
    --prompter_dropout 0.2  \
    --model_checkpoint $MODEL_PATH\