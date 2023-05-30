#! /bin/bash
PRED="/home/users/astar/i2r/stuait/scratch/Projects/disentangled_zero_dst/T5DST/src/save/t5REP_FOR_CR_SGD_pluto_global_local_prompt_do_0.2_p10_rept_noes_2ep_249_except_domain_none_slotlang_none_lr_0.0001_epoch_2_seed_249"
DSTC8="/home/users/astar/i2r/stuait/scratch/Projects/disentangled_zero_dst/T5DST/data/dstc8-schema-guided-dialogue"
EVAL_SET="test"

echo $PRED

python SGD_eval_script.py --prediction_dir $PRED --dstc8_data_dir $DSTC8 --eval_set $EVAL_SET --output_metric_file ${PRED}/"result.json"