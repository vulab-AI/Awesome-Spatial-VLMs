# Evaluation
In this section, we offer codes of the models we evaluated in the paper. Due to the different environments' requirements of each models, it's hard for us to offer a unified way to run all models in a code. As a result, we offer a spatial_eval.py for each model and installation instructions for each model or method.


### Commercial Models & General VLMs 

[Evaluation for Both](Commercial_General)


### Prompting Methods

[Installations for Prompting Methods](Train_Free_Promptings)

### Model-Centric Enchancement

[Installations for Model-Centric Enhancement](Model_Centric)

### Explicit 2D Information Injecting

[Installations for Explicit 2D Information Injecting](2D_Information)

### 3D Spatial Information Enhancement

[Installations for 3D Spatial Information Enhancement](3D_Information)

### Data-Centric Spatial Enhancement

[Installations for Data-Centric Spatial Enhancement](Data_Centric)


# Reference
If there are some problems or bug during installation, following are the original github links.
> **Note:** In order to unify and for ease of use, most the pip packages in our requirements.txt diff from the requirements in original github projects and we modify part of code to fit the evaluation. As a result, if you use `git clone ` through belowing link, please don't forget to copy the files in our github.

| Type | Models / Methods | Model Version | Model Source | Model Backbone | Multi-View |
|------|------------------|---------------|--------------|----------------|------------|
| **General Models** | | | | | |
| | GPT-4o | gpt-4o-2024-08-06 | [OpenAI](https://platform.openai.com/docs/models/gpt-4o) | – | ✓ |
| | GPT-5 | gpt-5-2025-08-07 | [OpenAI](https://platform.openai.com/docs/models/gpt-5) | – | ✓ |
| | Gemini 2.5 flash | gemini-2.5-flash | [Google Cloud](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash) | – | ✓ |
| | Gemini 2.5 pro | gemini-2.5-pro | [Google Cloud](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro) | – | ✓ |
| | Qwen2.5-VL-7B | Qwen/Qwen2.5-VL-7B-Instruct | [Huggingface](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | – | ✓ |
| | Qwen2.5-VL-72B | Qwen/Qwen2.5-VL-72B-Instruct | [Huggingface](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) | – | ✓ |
| | LLaVA-v1.5-7B | llava-hf/llava-1.5-7b-hf | [Huggingface](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | – | ✓ |
| | LLaVA-NeXT-7B | llava-hf/llava-v1.6-mistral-7b-hf | [Huggingface](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) | – | ✓ |
| | LLaVA-OneVision-7B | llava-hf/llava-onevision-qwen2-7b-ov-hf | [Huggingface](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf) | – | ✓ |
| | LLaVA-Next-72B | llava-hf/llava-next-72b-hf | [Huggingface](https://huggingface.co/llava-hf/llava-next-72b-hf) | – | ✓ |
| **General Models** | | | | | |
| | Spatialprompt | ---- | [Github](https://github.com/andrewliao11/Q-Spatial-Bench-code) | – | ✓ |
| | CCoT | ---- | [Github](https://github.com/chancharikmitra/CCoT) | – | ✓ |
| | Scaffold | ---- | [Github](https://github.com/THUNLP-MT/Scaffold) | – | ✓ |
| | SoM | ---- | [Github](https://github.com/microsoft/SoM) | – | ✓ |
| **Model-Centric Enhancement** | | | | | |
| | ROSS | HaochenWang/ross-qwen2-7b | [Huggingface](https://huggingface.co/HaochenWang/ross-qwen2-7b) | CLIP-ViT-L+Qwen2-7B | ✓ |
| | ViLaSR | inclusionAI/ViLaSR | [Huggingface](https://huggingface.co/inclusionAI/ViLaSR) | Qwen-2.5-VL-7B | ✓ |
| | M2-Reasoning-7B | inclusionAI/M2-Reasoning | [Huggingface](https://huggingface.co/inclusionAI/M2-Reasoning) | Qwen-2.5-VL-7B | ✓ |
| | LLaVA-AURORA | LLaVA-AURORA | [Github](https://github.com/mahtabbigverdi/Aurora-perception?tab=readme-ov-file) | LLaVA-v1.5-13B | ✓ |
| | AdaptVis | llava_1.5_adapt_vis | [Github](https://github.com/shiqichen17/AdaptVis) | LLaVA-v1.5-7B | ✓ |
| | Honeybee | Honeybee-C-7B-M256 | [Github](https://github.com/khanrc/honeybee) | CLIP ViT-L+Vicuna v1.5-7B | ✓ |
| | Cambrian-1 | nyu-visionx/cambrian-8b | [Huggingface](https://huggingface.co/nyu-visionx/cambrian-8b) | CLIP ViT-L+Vicuna-1.5-7B | ✓ |
| **Explicit 2D Information Injecting** | | | | | |
| | VPT | rp-yu/Qwen2-VL-7b-VPT-CLIP | [Huggingface](https://huggingface.co/rp-yu/Qwen2-VL-7b-VPT-CLIP) | Qwen2-VL-7B | ✓ |
| | VCoder | shi-labs/vcoder_llava-v1.5-7b | [Huggingface](https://huggingface.co/shi-labs/vcoder_llava-v1.5-7b) | LLaVA-v1.5-7B | ✓ |
| **3D Spatial Information Enhancement** | | | | | |
| | LLaVA-3D | ChaimZhu/LLaVA-3D-7B | [Huggingface](https://huggingface.co/ChaimZhu/LLaVA-3D-7B) | LLaVA-v1.5-7B | ✓ |
| | SpatialBot-3B | RussRobin/SpatialBot-3B | [Huggingface](https://huggingface.co/RussRobin/SpatialBot-3B) | Phi2-3B | ✗ |
| | VCoder (depth) | shi-labs/vcoder_ds_llava-v1.5-7b | [Huggingface](https://huggingface.co/shi-labs/vcoder_ds_llava-v1.5-7b) | Depth Encoder + LLaVA-v1.5-7B | ✓ |
| **Data-Centric Spatial Enhancement** | | | | | |
| | SpaceOm | remyxai/SpaceOm | [Huggingface](https://huggingface.co/remyxai/SpaceOm) | Qwen2.5VL-3B | ✓ |
| | SpaceQwen2.5-VL-3B-Instruct | remyxai/SpaceQwen2.5-VL-3B-Instruct | [Huggingface](https://huggingface.co/remyxai/SpaceQwen2.5-VL-3B-Instruct) | Qwen2.5-VL-3B | ✓ |
| | SpaceFlorence-2 | remyxai/SpaceFlorence-2 | [Huggingface](https://huggingface.co/remyxai/SpaceFlorence-2) | Florence-2-base | ✗ |
| | SpaceThinker-Qwen2.5VL-3B | remyxai/SpaceThinker-Qwen2.5VL-3B | [Huggingface](https://huggingface.co/remyxai/SpaceThinker-Qwen2.5VL-3B) | Qwen2.5-VL-3B | ✓ |
| | SpaceMantis | remyxai/SpaceMantis | [Huggingface](https://huggingface.co/remyxai/SpaceMantis) | Mantis-8B | ✓ |
| | SpaceLLaVA-13B | remyxai/SpaceLLaVA | [Huggingface](https://huggingface.co/remyxai/SpaceLLaVA) | LLaVA-v1.5-13B | ✓ |
| | SpaceLLaVA-1.5-7B | salma-remyx/spacellava-1.5-7b | [Huggingface](https://huggingface.co/salma-remyx/spacellava-1.5-7b) | LLaVA-v1.5-7B | ✓ |