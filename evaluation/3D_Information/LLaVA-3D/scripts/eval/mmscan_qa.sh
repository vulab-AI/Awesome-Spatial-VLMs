model_path=$1
question_path=$2
answer_path=$3
N=$4
chunk_id=$5

python llava/eval/model_mmscan_qa.py \
    --model-path "$model_path" \
    --question-file "$question_path" \
    --answers-file "$answer_path" \
    --num-chunks "$N" \
    --chunk-idx "$chunk_id" 