# Install Env
In this section, each model should build its env. All the codes run under **CUDA_12.1/CUDA_12.4**.

## VCoder
```
#create env
conda create -n vcoder python=3.10.18
conda activate vcoder
pip install -r requirements.txt

#in /VCoder
cd VCoder
python spatial_eval.py --output_path output/vcoder
```

## VPT
```
conda create -n vpt python=3.11.10
conda activate vpt
pip install -r requirements.txt

#in /VisualPerceptionToken
cd VisualPerceptionToken
python spatial_eval.py --output_path output/vpt
```

## Output
All the results will be in the `output/model_name`. Use the `metrics.ipynb` to calcualte the accuracy. Please adjust the regex pattern of output as needed.