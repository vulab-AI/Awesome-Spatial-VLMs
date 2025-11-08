python llava/eval/model_sqa3d.py \
        --model-path ./checkpoints/llava-3d-7b \
        --question-file playground/data/annotations/llava-3d-sqa3d_test_question.json \
        --answers-file ./llava-3d-7b-sqa3d_test_answer.json

python llava/eval/sqa3d_evaluator.py