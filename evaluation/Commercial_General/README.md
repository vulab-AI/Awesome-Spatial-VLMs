# Install Env for All
All the codes run under **CUDA_12.1/CUDA_12.4**.

```
#create env 
conda create -n vlm_3.11 python=3.11
conda activate vlm_3.11
pip install -r requirements.txt
```

# Commercial
## Gemini 
1. Under `/gemini_2_5`, get all data jsons and upload data to your gcloud
```
python gemini_data_preprocess.py
```

2. transfer jsons to jsonls, run the code in .ipynb

3. batch evaluation
```
python gemini_batch.py
```

## GPT
1.  under `/gpt`, get all data jsons
```
python gpt_data_preprocess_1.py
```
Then assumpt you have already upload your data on your google drive. (Batch api only support image_url that a link to donwload image.)

2. Get link from google drive
```
python gpt_data_preprocess_2.py
```
3.  Batch Evaluation
```
python gpt_batch.py
```

# Geenral Models
## Qwen2_5
```
python general.py --output_path output/qwen2_5_7b. --model_path   Qwen/Qwen2.5-VL-7B-Instruct
```
## LLava_1.5
```
python general.py --output_path output/llava_1_5_7b. --model_path  llava-hf/llava-1.5-7b-hf
```
## LLava_Next
```
python general.py --output_path output/llava_next_7b. --model_path  llava-hf/llava-v1.6-mistral-7b-hf
```
## LLava_Onevision
```
python general.py --output_path output/llava_onevision_7b. --model_path  llava-hf/llava-onevision-qwen2-7b-ov-hf
```

## Output
All the results will be in the `output/model_name`. Use the `metrics.ipynb` to calcualte the accuracy. Please adjust the regex pattern of output as needed.




