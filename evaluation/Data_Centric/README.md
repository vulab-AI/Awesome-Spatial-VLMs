# Install Env
In this section, each model should build its env. All the codes run under **CUDA_12.1/CUDA_12.4**.

# Install Env for All
All the codes run under **CUDA_12.1/CUDA_12.4**.

```
#create env 
conda create -n vlm_3.11 python=3.11
conda activate vlm_3.11
pip install -r requirements.txt
```

## SpaceOm
```
cd spaceom

python spaceom.py >spaceom.log 2>&1
```

## SpaceQwen
```
cd spaceqwen

python spaceqwen.py >spaceqwen.log 2>&1
```

## Space_Florence-2.0
```
cd space_florence2

python space_florence2_eval.py >spatial_florence2.log 2>&1
```
## SpaceThinker
```
cd spacethinker

python spacethinker_eval.py >spacethinker.log 2>&1
```

## SpaceMantis
```
cd spacemantis

python spacemantis_eval.py >spacemantis.log 2>&1
```

## SpaceLLaVA
7B:

```
cd spacellava

python spacellava7b_eval.py >spacellava7b.log 2>&1
```

13B:
```
cd spacellava/content
#Download ggml-model-q4_0.gguf  (from. https://huggingface.co/remyxai/SpaceLLaVA/tree/main)
wget https://huggingface.co/remyxai/SpaceLLaVA/resolve/main/ggml-model-q4_0.gguf?download=true
#Download mmproj-model-f16.gguf
wget https://huggingface.co/remyxai/SpaceLLaVA/resolve/main/ggml-model-f16.gguf?download=true
cd ..

#create env
conda create -n spacellava13b python=3.10
conda activate spacellava13b

# load modules on hpc or install manually
module load GCCore/13.3.0
module load CMake/3.24.4

# PyTorch (CUDA 12.1)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121

# llama-cpp-python 
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.20 --no-cache-dir

pip install datasets==2.14.0
pip install numpy==1.24.4
pip install Pillow==9.5.0
pip install tqdm==4.65.0

# Hugging Face
pip install huggingface-hub==0.16.4
pip install transformers==4.33.0

#start evaluation
python spacellava13b_eval.py >spacellava13b.log 2>&1
```

## Output
All the results will be in the `output/model_name`. Use the `metrics.ipynb` to calcualte the accuracy. Please adjust the regex pattern of output as needed.