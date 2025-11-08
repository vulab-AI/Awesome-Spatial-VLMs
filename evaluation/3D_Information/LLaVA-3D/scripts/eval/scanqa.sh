python llava/eval/model_scanqa.py \
        --model-path ./checkpoints/llava-3d-7b \
        --question-file playground/data/annotations/llava-3d-scanqa_val_question.json \
        --answers-file ./llava-3d-7b-scanqa_answer_val.json

python llava/eval/scanqa_evaluator.py