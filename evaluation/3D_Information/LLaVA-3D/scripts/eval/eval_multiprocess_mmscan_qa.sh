#!/bin/bash

# Assign the command line arguments to variables
model_path='checkpoints/llava-3d-7b'
question_path='playground/data/annotations/llava-3d-mmscan_QA_val.json'
base_answer_path='mmscan_qa'
N=6

# Loop over each chunk/process
for (( chunk_id=0; chunk_id<N; chunk_id++ ))
do
    # Define the answer path for each chunk
    answer_path="${base_answer_path}/result_${chunk_id}.json"
    if [ -f "$answer_path" ]; then
        rm "$answer_path"
    fi
    # Run the Python program in the background
    bash eval_single_process_mmscan_qa.sh ${model_path} ${question_path} ${answer_path} ${N} ${chunk_id}
    # Uncomment below if you need a slight delay between starting each process
    # sleep 0.1
done

# Wait for all background processes to finish
wait

merged_file="${base_answer_path}/result.json"
if [ -f "$merged_file" ]; then
    rm "$merged_file"
fi
# Merge all the JSONL files into one
#cat "${base_answer_path}"_*.jsonl > "${base_answer_path}.jsonl"
for ((i=0; i<N; i++)); do
  input_file="${base_answer_path}/result_${i}.json"
  cat "$input_file" >> "${base_answer_path}/result.json"
done
# remove the unmerged files
for (( chunk_id=0; chunk_id<N; chunk_id++ ))
do
    # Define the answer path for each chunk
    answer_path="${base_answer_path}/result_${chunk_id}.json"
    if [ -f "$answer_path" ]; then
        rm "$answer_path"
    fi
done