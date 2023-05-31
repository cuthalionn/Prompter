#! /bin/bash
PRED="PATH_TO_MODEL_DIR"
DSTC8="../data/dstc8-schema-guided-dialogue"
EVAL_SET="test"

echo $PRED

python SGD_eval_script.py --prediction_dir $PRED --dstc8_data_dir $DSTC8 --eval_set $EVAL_SET --output_metric_file ${PRED}/"result.json"