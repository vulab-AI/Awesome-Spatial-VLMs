# Install Env
In this section, each model should build its env. All the codes run under **CUDA_12.1/CUDA_12.4**.

## VCoder_depth
The code share the same env as VCoder in 2D_Information/VCoder.

```
#create env
conda create -n vcoder python=3.10.18
conda activate vcoder
pip install -r requirements.txt

#in /VCoder
cd VCoder
python spatial_eval_depth.py --output_path output/vcoder_depth
```

## LLaVA-3D
```
#create env
conda create -n llava-3d python=3.10.18
conda activate llava-3d
pip install -r requirements.txt

#in /LLaVA-3D
cd LLaVA-3D
python llava/eval/spatial_eval.py.py >LLaVA-3D.log 2>&1
```


## SpatialBot
```
#create env
conda create -n spatialbot python=3.11
conda activate spatialbot
pip install -r requirements.txt

#in /spatialbot
cd spatialbot
python spatial_eval.py >spatialbot.log 2>&1
```

## Output
All the results will be in the `output/model_name`. Use the `metrics.ipynb` to calcualte the accuracy. Please adjust the regex pattern of output as needed.