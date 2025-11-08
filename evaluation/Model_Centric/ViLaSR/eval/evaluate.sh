#!/bin/bash
script_paths=(
"vsi_bench"
"maze"
"SpatialEval_spatialreal"
"spar_bench"
"mmsi_bench"
)

MODE=rl  # zero_shot, cold_start, reflective, rl
echo START Score Answer...

for ((i=0; i<${#script_paths[@]}; i++)); do
    QUESTION_FILE=${script_paths[i]}

    # mkdir -p $RESULTDIR/scores/$MODE/$QUESTION_FILE
    RESULTDIR=./eval/results/$MODE/${QUESTION_FILE}
    echo ""
    echo "#----------------------------------------#"
    echo "Processing dataset: ${QUESTION_FILE}"
    echo "Results path: ${RESULTDIR}/results.jsonl" 
    echo "#----------------------------------------#"
    python eval/evaluate.py \
        --dataset $QUESTION_FILE \
        --question-file $RESULTDIR/results.jsonl \
        --result-file $RESULTDIR/results.jsonl \
        --output-result $RESULTDIR/scores.jsonl 
    echo "Evaluation complete for ${QUESTION_FILE}"
    echo ""
done

echo "===========================================" 
echo "Evaluation Complete!"
echo "===========================================" 
