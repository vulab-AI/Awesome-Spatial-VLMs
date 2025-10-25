# Install Env
In this section, each model should build its env. All the codes run under **CUDA_12.1/CUDA_12.4**.

## ViLaSR
```
#create env
conda create -n vilasr python=3.10
conda activate vilasr
pip install -r requirements.txt

#in /ViLaSR
cd ViLaSR
python spatial_eval.py >ViLaSR.log 2>&1
```

## Ross
```
#create env
conda create -n ross python=3.10
conda activate ross
pip install -r requirements.txt

#in /ross
cd ross
python spatial_eval.py >ross.log 2>&1
```

## Honeybee
```
#create env
conda create -n honeybee python=3.10
conda activate honeybee
pip install -r requirements.txt

#in /honeybee
cd honeybee
python spatial_eval.py >honeybee.log 2>&1
```

## Cambrian
```
#create env
conda create -n cambrian python=3.10
conda activate cambrian
pip install -r requirements.txt

#in /cambrian
cd cambrian
python spatial_eval.py >cambrian.log 2>&1
```

## LLaVA-AURORA
```
#create env
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip 
pip install -e .
pip install -e ".[train]"
pip install peft==0.11.1
pip install flash-attn==2.5.9.post1

#download checkpoint
cd LLaVA-Aurora
wget https://drive.google.com/file/d/1r7WYQWYA6VDpzfxPIHP1zEUgBYQmwNIj/view?usp=sharing
unzip checkpoints.zip

#start evaluation
python LLaVA/llava/eval/spatial_eval.py >llava-3d.log 2>&1
```

## AdaptVis 
```
#create env
conda create -n adaptvis python=3.10
conda activate adaptvis
pip install -r requirements.txt

#in /AdaptVis
cd AdaptVis
python spatial_eval.py >AdaptVis.log 2>&1
```

## M2-Reasoning
```
#create env
conda create -n m2_reasoning python=3.11
conda activate m2_reasoning
pip install -r requirements.txt

#in /m2_reasoning
cd m2_reasoning
python spatial_eval.py >m2_reasoning.log 2>&1
```

## Output
All the results will be in the `output/model_name`. Use the `metrics.ipynb` to calcualte the accuracy. Please adjust the regex pattern of output as needed.

 