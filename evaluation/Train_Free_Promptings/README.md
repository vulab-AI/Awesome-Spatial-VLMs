# Install
All the codes run under **CUDA_12.1/CUDA_12.4**.

```
#create env 
conda create -n vlm_3.11 python=3.11
conda activate vlm_3.11
pip install -r requirements.txt
```

## Compositional Chain of Thoughts (CCoT)
The main idea of CCoT is to generate a graph of image first and then augment the next prompt input to answer the question.
```
python spatial_eval.py --model_path qwen/Qwen2.5-VL-7B-Instruct --output_path output/qwen2_5_7b_ccot --prompt_type ccot > qwen2_5_7b_ccot.log 2>&1
```

## SpatialPrompt
The main idea of SpatialPrompt is ask model to find a referred object while thinking and answering the question.
```
python spatial_eval.py --model_path qwen/Qwen2.5-VL-7B-Instruct --output_path output/qwen2_5_7b_spatialprompt --prompt_type spatialprompt > qwen2_5_7b_spatialprompt.log 2>&1
```

## Scofford
The main idea of Scofford is to build axis coordinate on the image first, and then input augmented image with prompt to answer the question.
```
python spatial_eval.py --model_path qwen/Qwen2.5-VL-7B-Instruct --output_path output/qwen2_5_7b_scaffold --prompt_type scaffold > qwen2_5_7b_scaffold.log 2>&1
```

## Set of Masks (SoM)
The main idea of Som is to get a automatically segmentation set of the whole image and then input the annotated images (masks on it) with prompt to answer the question.

Unlike the other 3 methods, SoM needs to generate masks of images first. In `/som`, You should follow the [Som](https://github.com/microsoft/SoM) to install segmentation packages if the following commands doesn't work.

```
#create environment
conda create -n seem python=3.10.18
conda activate seem 

#You should cd /som first.
pip install -r requirement.txt

#on HPC or install manually by apt install
module load GEOS/3.10.3-GCC-11.3.0
module load OpenMPI/4.1.4-GCC-11.3.0

# install SEEM
pip install git+https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git@package
# install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git
# install Semantic-SAM
pip install git+https://github.com/UX-Decoder/Semantic-SAM.git@package
# install Deformable Convolution for Semantic-SAM
cd ops && bash make.sh && cd ..

# common error fix:
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'

#Download pretrained models.
sh download_ckpt.sh
```

Then get mask of spatial evaluation benchmark.
```
#download data from huggingface to local first
conda activate vlm_3.11
python download_dataset2img.py

# data will save under /som
cd som
conda activate seem
python mask.py > mask.log 2>&1
```

Aftet get all masks of images.
```
python spatial_eval.py --model_path qwen/Qwen2.5-VL-7B-Instruct --output_path output/qwen2_5_7b_som  --prompt_type som > qwen2_5_7b_som.log 2>&1
```
## Output
All the results will be in the `output/model_name`. Use the `metrics.ipynb` to calcualte the accuracy. Please adjust the regex pattern of output as needed.


