  <h1 align="center" style="font-size: 1.7rem">Awesome Spatial VLMs</h1>
<!-- 
  <p align="center">
    <a href="https://scholar.google.com/citations?user=xlIBwREAAAAJ&hl=en">Disheng Liu</a>,
    <a href="https://jiagengliu02.github.io/">Tuo Liang</a>,
    <a href="https://scholar.google.com/citations?user=oV8sqb0AAAAJ&hl=zh-CN">Zhe Hu</a>,
    <a href="https://scholar.google.com/citations?user=7CLFLX0AAAAJ&hl=en">Jierui Peng</a>,
    <a href="https://yiren-lu.com/">Yiren Lu</a>,
    <a href="https://sites.google.com/view/homepage-of-yi-xu">Yi Xu</a>,
    <a href="https://www1.ece.neu.edu/~yunfu/">Yun Fu</a>,
    <a href="https://yin-yu.github.io/">Yu Yin</a>
  </p> -->

> A curated list of resources for Spatial Intelligence in Vision-Language Models.

This repository is the official, community-maintained resource for the survey paper: **Spatial Intelligence in Vision-Language Models: A Comprehensive Survey**

  <p align="center">
    <a href="https://github.com/vulab-AI/Awesome-Spatial-VLMs/blob/main/Spatial_VLM_survey.pdf">
      <img src="https://img.shields.io/badge/Paper-PDF-blue?style=flat&logo=google-scholar&logoColor=blue" alt="Paper PDF">
    </a>
    <a href='https://huggingface.co/datasets/LLDDSS/Awesome_Spatial_VQA_Benchmarks' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Evaluated-Dataset-yellow?style=flat&logo=huggingface&logoColor=yellow' alt='Evaluated Data'>
    </a>
    <a href='https://github.com/vulab-AI/Awesome-Spatial-VLMs/blob/main/evaluation/README.md' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Evaluation-Code-black?style=flat&logo=github&logoColor=black' alt='Evaluated Data'>
    </a>
  </p>

🤝 This repository will be continuously updated, and we warmly invite contributions. **If you have a paper, dataset, or model to add, please submit a pull request or open an issue for discussion.**


## Table of Contents
- [Overview](#overview)
- [Awesome Papers](#📚-awesome-papers)
  - [Training-Free Prompting](#training-free-prompting)
  - [Model-Centric Enhancements](#model-centric-enhancements)
  - [Explicit 2D Information Injection](#explicit-2d-information-injection)
  - [3D Information Enhancement](#3d-information-enhancement)
  - [Data-Centric Spatial Enhancement](#data-centric-spatial-enhancement)
- [Datasets and Benchmarks](#📚-datasets-and-benchmarks)
  - [Training Corpora](#spatially-oriented-training-corpora)
  - [Evaluation Benchmarks](#evaluation-benchmarks)
- [Spatial VLM Leaderboard & Evaluation Toolkit](#🏆-spatial-vlm-leaderboard--evaluation-toolkit)
  - [Leaderboard](#leaderboard)
  - [Evaluation Toolkit](#evaluation-toolkit)
- [Related Surveys](#related-surveys)
- [Citation](#citation)


## Overview
This repository uses the framework from our survey paper to systematically organize the field of Spatial Intelligence in VLMs.
- **The “What”: A Cognitive Hierarchy** 🧩  
  We define spatial intelligence as a 3-level hierarchy, and group tasks, datasets, and benchmarks by required capability:  
  **L1** *Perception* of intrinsic 3D attributes (e.g., size, orientation) &rarr; **L2** relational *Understanding* &rarr; **L3** *Extrapolation* (e.g., hidden-state inference, future prediction).
- **The “How”: A Taxonomy of Methods** 🚀  
  Methods are organized into five families, giving you a clear map of the current landscape. See details in [Awesome Papers](#awesome-papers).
- **Where We Are: Evaluation Results and Toolkit** 🏆  
  See how current models perform! 
  - **Standardized Leaderboard:** We report results for **37+ VLMs** across all L1/L2/L3 tasks.
  - **Open Evaluation Toolkit:** Reproduce our protocols and **evaluate your own models** under the same settings.

<div align='center'><img src="./samples/outline.jpg"  alt="Overview Diagram" width="85%"/></div>

  <!-- - **L1: Spatial Perception:** Recognizing individual objects and their intrinsic 3D attributes (e.g., size, orientation, 3D segmentation).
  - **L2: Spatial Understanding:** Reasoning about the extrinsic, relational properties among multiple objects (e.g., "the dog to the left of the cat").
  - **L3: Spatial Extrapolation:** Inferring hidden states, predicting future configurations, or reasoning from a situated perspective (e.g., mental rotation, pathfinding). -->


## 🚀 Awesome Papers
### Training-Free Prompting

Author et al., Title (Venue YYYY). [paper] [code] — {method tag}; targets {L1/L2/L3}.

| Textual Prompting Methods | Institution | Venue| Code | Model Zoo |
|-------|-------------|------|------|------------|
| Reasoning Paths with Reference Objects Elicit Quantitative Spatial Reasoning in Large Vision-Language Models | University of Toronto | EMNLP2024  | [Link](https://github.com/andrewliao11/Q-Spatial-Bench-code) |  |
| Compositional Chain-of-Thought Prompting for Large Multimodal Models | University of California, Berkeley | CVPR2024  | [Link](https://github.com/chancharikmitra/CCoT) |  |
| Enhancing the Spatial Awareness Capability of Multi-Modal Large Language Model | Peking University | CoRR2023  |  |  |
| SpatialPIN: Enhancing Spatial Reasoning Capabilities of Vision-Language Models through Prompting and Interacting 3D Priors | University of Oxford | NeurIPS2024  |  |  |
| SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation | Tsinghua University | CoRR 2025  | [Link](https://github.com/qizekun/SoFar) |  |
--------


| Visual Prompting Methods | Institution | Venue| Code | Checkpoint |
|-------|-------------|------|------|------------|
| Fine-Grained Visual Prompting | Nanjing University of Science and Technology | NeurIPS2023  | [Link](https://github.com/ylingfeng/FGVP) |  |
| Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V | Microsoft Research, Redmond | arXiv2023  | [Link](https://github.com/microsoft/SoM) |  |
| 3daxisprompt: Promoting the 3d grounding and reasoning in gpt-4o | Shanghai AI Lab | Neurocomp’2025  |  |  |
| Coarse correspondences boost spatial-temporal reasoning in multimodal language model | University of Washington | CVPR2025  |  |  |
| SpatialPrompting: Keyframe-driven Zero-Shot Spatial Reasoning with Off-the-Shelf Multimodal Large Language Models | Toyota Central R&D Labs | arXiv2025  |  |  |
| Mindjourney: Test-time scaling with world models for spatial reasoning | UMass Amherst | NeurIPS2025  | [Link](https://github.com/UMass-Embodied-AGI/MindJourney) |  |
--------

| Hybrid Prompting | Institution | Venue| Code | Checkpoint |
|-------|-------------|------|------|------------|
| Seeground: See and ground for zero-shot open-vocabulary 3d visual grounding | HKUST(Guangzhou) | CVPR 2025  | [Link](https://github.com/iris0329/SeeGround) |  |
| Scaffolding coordinates to promote vision-language coordination in large multi-modal models | Tsinghua University | COLING2025  | [Link](https://github.com/THUNLP-MT/Scaffold) |  |
| Mind’s Eye of LLMs: Visualization-of-Thought Elicits Spatial Reasoning in Large Language Models | Microsoft Research | NeurIPS2024  | [Link](https://github.com/microsoft/visualization-of-thought/) |  |
| Image-of-Thought Prompting for Visual Reasoning Refinement in Multimodal Large Language Models | Westlake University | arXiv2024  |  |  |
| Visual sketchpad: Sketching as a visual chain of thought for multimodal language models | University of Washington | NeurIPS2024  | [Link](https://github.com/Yushi-Hu/VisualSketchpad) |  |
--------

### Model-Centric Enhancements
<!-- Advanced Training Strategies -->

| Title | Institution | Venue| Code | Checkpoint |
|-------|-------------|------|------|------------|
| ROSS | Institute of Automation, Chinese Academy of Sciences |  ICLR 2025  | [Link](https://github.com/haochen-wang409/ross) | [Link](https://huggingface.co/HaochenWang/ross-qwen2-7b) |
| SpatialCoT | Huawei Noah's Ark Lab | arXiv2025  |  |  |
| Cube-LLM | UT Austin |  ICLR 2025  |  |  |
| MVoT | Microsoft Research | ICML2025  | [Link](https://github.com/chengzu-li/MVoT) |  |
| SpaceR | Peking University | arXiv2025  | [Link](https://github.com/OuyangKun10/SpaceR?tab=readme-ov-file) | [Link](https://huggingface.co/RUBBISHLIKE/SpaceR) |
| ViLaSR | Institute of Automation, Chinese Academy of Sciences. | arXiv2025  | [Link](https://github.com/AntResearchNLP/ViLaSR) | [Link](https://huggingface.co/inclusionAI/ViLaSR/tree/main) |
| M2-Reasoning | Inclusion AI, Ant Group | arXiv2025  | [Link](https://github.com/inclusionAI/M2-Reasoning) | [Link](https://huggingface.co/inclusionAI/M2-Reasoning) |
| Svqa-r1: Reinforcing spatial reasoning in mllms via view-consistent reward optimization, | Stony Brook University | arXiv2025  |  |  |
| Spatialreasoner: Towards explicit and generalizable 3d spatial reasoning, | Johns Hopkins University | NeurIPS 2025  | [Link](https://github.com/johnson111788/SpatialReasoner) | [Link](https://huggingface.co/collections/ccvl/spatialreasoner-68114caec81774edbf1781d3) |
| Embodied-r: Collaborative framework for activating embodied spatial reasoning in foun2339 dation models via reinforcement learning | Tsinghua University | arXiv2025  | [Link](https://github.com/EmbodiedCity/Embodied-R.code) | [Link](https://huggingface.co/EmbodiedCity/Embodied-R) |
| Spatialladder: Progressive training for spatial reasoning in vision-language models | Zhejiang University | arXiv2025  | [Link](https://github.com/zju-real/SpatialLadder) | [Link](https://huggingface.co/hongxingli/SpatialLadder-3B) |
| Spacer: Reinforcing mllms in video spatial reasoning | Peking University | arXiv2025  | [Link](https://github.com/OuyangKun10/SpaceR) | [Link](https://huggingface.co/RUBBISHLIKE/SpaceR) |
| Fine-Grained Preference Optimization Improves Spatial Reasoning in VLMs | UIUC | NeurIPS 2025  |  |  |
| Metaspatial: Reinforcing 3d spatial reasoning in vlms for the metaverse | Northwestern University | arXiv2025  | [Link](https://github.com/PzySeere/MetaSpatial) |  |
| LOCALITY ALIGNMENT IMPROVES VISION-LANGUAGEMODELS | Stanford University | ICLR2025  |  |  |
| What makes for good visual tokenizers for large language models | National University of Singapore 2ARC Lab, | arXiv2023  | [Link](https://github.com/TencentARC/GVT) | [Link](https://github.com/TencentARC/GVT/tree/master/gvt) |
| Perception Tokens Enhance Visual Reasoning in Multimodal Language Models | University of Washington | CVPR 2025  | [Link](https://github.com/mahtabbigverdi/Aurora-perception) | [Link](https://drive.google.com/file/d/1r7WYQWYA6VDpzfxPIHP1zEUgBYQmwNIj/view) |
--------

#### Architectural Enhancements

| Title | Institution | Venue| Code | Checkpoint |
|-------|-------------|------|------|------------|
| To Preserve or To Compress: An In-Depth Study of Connector Selection in Multimodal Large Language Models | Digital Twin Institute, Eastern Institute of Technology, Ningbo, China | EMNLP2024  | [Link](https://github.com/EIT-NLP/Connector-Selection-for-MLLM) |  |
| Honeybee: Locality enhanced projector for multimodal llm | Kakao Brain | CVPR2024 (Highlight)  | [Link](https://github.com/khanrc/honeybee) | [Link](https://github.com/khanrc/honeybee) |
| Cambrian-1: A fully open, vision-centric exploration of multimodal llm | New York University | NeurIPS 2024 (Oral)  | [Link](https://github.com/cambrian-mllm/cambrian) | [Link](https://huggingface.co/collections/nyu-visionx/cambrian-1-models-666fa7116d5420e514b0f23c) |
| Why is spatial reasoning hard for vlms? an attention mechanism perspective on focus areas | City University of Hong Kong | ICML2025  | [Link](https://github.com/shiqichen17/AdaptVis) | [Link](https://github.com/shiqichen17/AdaptVis) |
| Contrastive region guidance: Improving grounding in vision-language mod2381 els without training | unc | ECCV 2024   | [Link](https://github.com/meetdavidwan/crg) | [Link](https://github.com/meetdavidwan/crg) |
--------

#### Encoder-Level Improvements

| Title | Institution | Venue| Code | Checkpoint |
|-------|-------------|------|------|------------|
| Spatialclip: Learning 3d-aware image representations from spatially discriminative language, | Zhejiang University | CVPR2025  | [Link](https://github.com/SpatialVision/Spatial-CLIP) |  |
| Introducing Visual Perception Token into Multimodal Large Language Model | National University of Singapore | arXiv2025  | [Link](https://github.com/yu-rp/VisualPerceptionToken?tab=readme-ov-file) | [Link](https://huggingface.co/collections/rp-yu/vpt-models-67b6afdc8679a05a2876f07a) |
| Prismatic vlms: Investigating the design space of visually-conditioned language models | Stanford | ICML 2024  | [Link](https://github.com/TRI-ML/prismatic-vlms?tab=readme-ov-file#pretrained-models) | [Link](https://github.com/TRI-ML/prismatic-vlms) |
| Incorporating Visual Experts to Resolve the Information Loss in Multimodal Large Language Models | Huawei | IJCAI2025  |  |  |
| Spatialllm | Johns Hopkins University | CVPR2025  |  |  |
| Cambrian-1 | New York University | NeurIPS 2024 (Oral)  | [Link](https://github.com/cambrian-mllm/cambrian) | [Link](https://huggingface.co/collections/nyu-visionx/cambrian-1-models-666fa7116d5420e514b0f23c) |
| ARGUS | University of Illinois Urbana-Champaign | CVPR2025  |  |  |
| Eagle 2 | NVIDIA | ICLR 2025  | [Link](https://github.com/NVlabs/EAGLE?tab=readme-ov-file) | [Link](https://huggingface.co/nvidia/Eagle2.5-8B) |
--------

### Explicit 2D Information Injection
#### Object Region Guidance

| Title | Institution | Venue| Code | Checkpoint |
|-------|-------------|------|------|------------|
| RegionGPT  | Nvidia | CVPR2024  |  |  |
| GPT4RoI | The University of Hong Kong | ECCV 2024 Workshops  | [Link](https://github.com/jshilong/GPT4RoI?tab=readme-ov-file) | [Link](https://huggingface.co/shilongz/GPT4RoI-7B-delta-V0) |
| PVIT | Tsinghua University | CoRR 2023  | [Link](https://github.com/PVIT-official/PVIT?tab=readme-ov-file#pvit-weights) | [Link](https://huggingface.co/PVIT/pvit) |
| VCoder | Georgia Tech | CVPR2024  | [Link](https://github.com/SHI-Labs/VCoder) | [Link](https://huggingface.co/models?search=vcoder) |
| Argus: Vision-centric reasoning with grounded chain-of-thought | University of Illinois Urbana-Champaign | CVPR2025  |  |  |
| CoVLM | UMass Amhers | ICLR 2024  | [Link](https://github.com/UMass-Embodied-AGI/CoVLM?tab=readme-ov-file) | [Link](https://github.com/UMass-Embodied-AGI/CoVLM?tab=readme-ov-file) |
| PEVL | Tsinghua University | EMNLP 2022   | [Link](https://github.com/thunlp/PEVL) | [Link](https://github.com/thunlp/PEVL) |
| Learning to Localize Objects Improves Spatial Reasoning in Visual-LLMs | Meta | CVPR2024  |  |  |
| Lyrics | International Digital Economy Academy | arXiv2025  |  |  |
--------

### Explicit spatial relationship

| Title | Institution | Venue| Code | Checkpoint |
|-------|-------------|------|------|------------|
| SGVL | Tel-Aviv University | ACL2023  | [Link](https://github.com/AlonMendelson/SGVL) | [Link](https://drive.google.com/file/d/13jzpcLgGalO3hkiqVwziNAlCEZD90ENN/view) |
| Object-centric Binding in Contrastive Language-Image Pretraining | meta | arXiv2025  |  |  |
| Seeing Beyond the Scene: Enhancing Vision-Language Models with Interactional Reasoning | South China University of Technology | arXiv2025  |  |  |
| LLaVA-SG |  Tsinghua University | arXiv2025  |  |  |
--------

### 3D Information Enhancement
#### Explicit 3D Geometric Representations

| Title | Institution | Venue| Code | Checkpoint |
|-------|-------------|------|------|------------|
| Spatial 3d-llm: Progressive spatial awareness for advanced 3d vision-language understanding | Beijing Digital Native Digital City Research Center | arXiv2025  |  |  |
| Segpoint: Segment any point cloud via large language model | Nanyang Technological University | ECCV2024  |  |  |
| Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning | Fudan University | CVPR2024  | [Link](https://github.com/Open3DA/LL3DA) |  |
| Scanqa: 3d question answering for spatial scene understanding | Kyoto Universit | CVPR2022  | [Link](https://github.com/ATR-DBI/ScanQA) |  |
| 3d-llm: Injecting the 3d world into large language models | UCLA | NeurIPS2023  | [Link](https://github.com/UMass-Embodied-AGI/3D-LLM) |  |
| 3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer | The University of Adelaide | CVPR2025  | [Link](https://github.com/djiajunustc/3D-LLaVA?tab=readme-ov-file) | [Link](https://huggingface.co/djiajunustc/3D-LLaVA-7B-LoRA) |
| Situational Awareness Matters in 3D Vision Language Reasoning | UIUC | CVPR 2024  | [Link](https://github.com/YunzeMan/Situation3D) |  |
| Lscenellm: Enhancing large 3d scene understanding using adaptive visual preferences | South China University of Technology | CVPR 2025  | [Link](https://github.com/Hoyyyaard/LSceneLLM) | [Link](https://huggingface.co/Hoyard/LSceneLLM) |
| Vcoder: Versatile vision encoders for  multimodal large language model | Georgia Tech | CVPR2024  | [Link](https://github.com/SHI-Labs/VCoder) | [Link](https://huggingface.co/models?search=vcoder) |
| SpatialBot: Precise Spatial Understanding with Vision Language Models | Shanghai Jiao Tong University | ICRA2025  | [Link](https://github.com/BAAI-DCAI/SpatialBot) | [Link](https://huggingface.co/RussRobin/SpatialBot-3B) |
| SpatialRGPT: Grounded Spatial Reasoning in Vision-Language Models | UC San Diego | NeurIPS2024  | [Link](https://github.com/AnjieCheng/SpatialRGPT) | [Link](https://huggingface.co/collections/a8cheng/spatialrgpt-grounded-spatial-reasoning-in-vlms-66fef10465966adc81819723) |
| SmolRGPT: Efficient Spatial Reasoning for Warehouse Environments with 600M Parameters | Universit ́e de Moncton | ICCVW'25  | [Link](https://github.com/abtraore/SmolRGPT) | [Link](https://huggingface.co/collections/Abdrah/smolrgpt-checkpoints-6893bad56127440ef250486e) |
| RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics | Beihang University | NeurIPS2025  | [Link](https://github.com/Zhoues/RoboRefer) | [Link](https://huggingface.co/collections/Zhoues/roborefer-and-refspatial-6857c97848fab02271310b89) |
| SSR: Enhancing Depth Perception in Vision-Language Models via Rationale-Guided Spatial Reasoning | Westlake University | NeurIPS 2025  | [Link](https://github.com/yliu-cs/SSR) | [Link](https://huggingface.co/collections/yliu-cs/ssr-682d44496b64e4edd94092bb) |
| SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models | Zhejiang University | NeurIPS2025  | [Link](https://github.com/cpystan/SD-VLM) | [Link](https://huggingface.co/cpystan/SD-VLM-7B) |
--------

#### Implicit 3D from Egocentric Views

| Title | Institution | Venue| Code | Checkpoint |
|-------|-------------|------|------|------------|
| VLM-3R: Vision-Language Models Augmented ·with Instruction-Aligned 3D Reconstruction | UT Austin | arXiv2025  | [Link](https://github.com/VITA-Group/VLM-3R) |  |
| Spatial-MLLM: Boosting MLLM Capabilities in Visual-based Spatial Intelligence | Tsinghua University | arXiv2025  | [Link](https://github.com/diankun-wu/Spatial-MLLM) |  |
| SplatTalk: 3D VQA with Gaussian Splatting | Georgia Institute of Technology | ICCV 2025  | [Link](https://splat-talk.github.io/) |  |
| I Know About “Up”! Enhancing Spatial Reasoning in Visual Language Models Through 3D Reconstruction | Guangdong Polytechnic Normal Universit | arXiv2024  |  |  |
--------

#### Scene-level information + Ego-centric

| Title | Institution | Venue| Code | Checkpoint |
|-------|-------------|------|------|------------|
| LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities | The University of Hong Kong | ICCV 2025  | [Link](https://github.com/ZCMax/LLaVA-3D) | [Link](https://huggingface.co/ChaimZhu/LLaVA-3D-7B) |
| 3D Concept Learning and Reasoning from Multi-View Images | UCLA | CVPR2023  | [Link](https://github.com/evelinehong/3D-CLR-Official) |  |
| Gpt4scene: Understand 3d scenes from videos with vision-language models | The University of Hong Kong | arXiv2025  | [Link](https://github.com/Qi-Zhangyang/GPT4Scene-and-VLN-R1) | [Link](https://huggingface.co/alexzyqi/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512) |
| Scene-LLM: Extending Language Model for 3D Visual Understanding and Reasoning | Brown University | WACV2025  |  |  |
| Chat-scene: Bridging 3d scene and large language models with object identifiers | Zhejiang University | NeurIPS2024  | [Link](https://github.com/ZzZZCHS/Chat-Scene) |  |
| Robin3D: Improving 3D Large Language Model via Robust Instruction Tuning | University of Illinois Chicago | ICCV 2025  | [Link](https://github.com/WeitaiKang/Robin3D?tab=readme-ov-file) | [Link](https://drive.google.com/drive/folders/14Si8bdWI3N5NEeVDLhmAlxilWPl0f_Wp?usp=sharing) |
| Inst3D-LMM: Instance-Aware 3D Scene Understanding with Multi-modal Instruction Tuning | Zhejiang University | CVPR2025  | [Link](https://github.com/hanxunyu/Inst3D-LMM) |  |
| DSPNet: Dual-vision Scene Perception for Robust 3D Question Answering | Sun Yat-sen University, | CVPR2025  | [Link](https://github.com/LZ-CH/DSPNet) | [Link](https://github.com/LZ-CH/DSPNet) |
| An Embodied Generalist Agent in 3D World | Beijing Institute for General Artificial Intelligence (BIGAI) | ICML2024  | [Link](https://github.com/embodied-generalist/embodied-generalist) | [Link](https://huggingface.co/datasets/huangjy-pku/LEO_data/tree/main) |
| ScanReason: Empowering 3D Visual Grounding with Reasoning Capabilities | The University of Hong Kong | ECCV2024  | [Link](https://github.com/ZCMax/ScanReason) |  |
| MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs | Apple | ICCV 2025  | [Link](https://github.com/apple/ml-cubifyanything) |  |
--------

### Data-Centric Spatial Enhancement
#### Manifesting Spatial Relations in 2D Images

| Title | Institution | Venue| Code | Checkpoint |
|-------|-------------|------|------|------------|
| Position-Enhanced Visual Instruction Tuning for Multimodal Large Language Models | Tsinghua University | arXiv23  | [Link](https://github.com/PVIT-official/PVIT) | [Link](https://huggingface.co/PVIT/pvit) |
| SpaRE | University of Waterloo | arXiv25  |  |  |
| The all-seeing project v2: Towards general relation comprehension of the open world | Shanghai AI Laboratory | ECCV24  | [Link](https://github.com/OpenGVLab/all-seeing?tab=readme-ov-file) | [Link](https://huggingface.co/OpenGVLab/ASMv2) |
| Kosmos-2  | Microsoft Research | ICLR24  | [Link](https://github.com/microsoft/unilm/tree/master/kosmos-2) | [Link](https://huggingface.co/microsoft/kosmos-2-patch14-224) |
| Pseudo-Q | Tsinghua University | CVPR22  | [Link](https://github.com/LeapLabTHU/Pseudo-Q?tab=readme-ov-file) |  |
--------

#### Manifesting Spatial Priors in 3D and Synthetic Worlds

| Title | Institution | Venue| Code | Checkpoint |
|-------|-------------|------|------|------------|
| SpatialVLM | Google DeepMind | CVPR24  | [Link](https://spatial-vlm.github.io/#community-implementation) | [Link](https://github.com/remyxai/VQASynth?tab=readme-ov-file#models-trained-using-vqasynth-) |
| LLaVA-SpaceSGG | City University of Hong Kong | WACV25  | [Link](https://github.com/Endlinc/LLaVA-SpaceSGG?tab=readme-ov-file) | [Link](https://huggingface.co/wumengyangok/LLaVA-SpaceSGG/tree/main) |
| RoboSpatial | NVIDIA | CVPR25  | [Link](https://github.com/NVlabs/RoboSpatial) |  |
| SPARTUN3D | Michigan State University & UC Davis | ICLR25  |  |  |
| MSQA | BIGAI | NeurIPS24  | [Link](https://github.com/MSR3D/MSR3D) |  |
| MultiSPA | Meta FAIR | arXiv25  | [Link](https://github.com/facebookresearch/Multi-SpatialMLLM?tab=readme-ov-file#-model-training) |  |
| Sparkle | Massachusetts Institute of Technology | arXiv25  |  |  |
| Orient Anything | Zhejiang University  | ICML25  | [Link](https://github.com/SpatialVision/Orient-Anything?tab=readme-ov-file) | [Link](https://huggingface.co/Viglong/Orient-Anything/blob/main/croplargeEX2/dino_weight.pt) |


## 📚 Datasets and Benchmarks
> A comprehensive list of datasets for training and evaluation.

### Spatially-Oriented Training Corpora

<!-- | Datasets | Venue  | Perc. | Unders. | Extrap. | Task| Size | Modality|
|-------|-------------|------|------|------------|
| [Proximity-110K](https://huggingface.co/Electronics/ProximityQA/blob/main/llava_proximity-mix.json) | [ArXiv2024]()  | [Link](https://github.com/AlonMendelson/SGVL) | [Link](https://drive.google.com/file/d/13jzpcLgGalO3hkiqVwziNAlCEZD90ENN/view) |

&  & \ding{51} & &  & depth estimation & 989,877 & Visual Genome, COCO &  RGB \\ -->

### Evaluation Benchmarks


## 🏆 Spatial VLM Leaderboard & Evaluation Toolkit

### Leaderboard
- **Leaderboard:** standardized results across representative datasets; submit via PR.

### Evaluation Toolkit
- **Tooling:** loaders, evaluation scripts, and metric definitions for spatial tasks.

To provide a clear performance baseline and facilitate future research, we provide our full evaluation toolkit. This suite includes the comprehensive results from our TPAMI survey, the open-source code to reproduce our evaluation, and the 9 integrated benchmarks hosted on Hugging Face.

For nine widely used spatial benchmarks, we provide:
- **Results:** reported performance of representative VLMs.
- **Evaluation code:** scripts to reproduce the metrics.
- **Benchmark packages:** integrated Hugging Face versions of each benchmark for plug-and-play evaluation.

These leaderboards are intended to make comparison across models transparent and reproducible.

Quick Links
- The script to run the evaluation.

- Download all 9 integrated benchmarks in one unified format.

- See the full results from our paper.

### 🏆 Main Leaderboard

The table below presents the main results from our survey, comparing 38 models across 9 benchmarks. Scores are QA Accuracy (%). Benchmarks are grouped by our Cognitive Hierarchy.

We invite the community to benchmark new models using our suite. Please to add your model's results!


### 🧑‍🔬 How to Evaluate Your Model

Facilitating the evaluation of published spatial related benchmarks, we summarize the dataset used in the evaluation section.

The related code is stored [here](evaluation/README.md).



We recollect the published spatial related datasets for evaluation. The following table summarizes the key datasets used for benchmarking spatial vision-language models:

<table>
  <tr>
    <th>Dataset Name</th>
    <th>Description</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>EgoOrientBench</td>
    <td>Egocentric spatial understanding benchmark</td>
    <td rowspan="10"><a href="https://huggingface.co/datasets/LLDDSS/Awesome_Spatial_VQA_Benchmarks">Link</a></td>
  </tr>
  <tr><td>GeoMeter(real)</td><td>A depth-aware spatial reasoning benchmark</td></tr>
  <tr><td>SEED-Bench (Spatial section)</td><td>Subset focusing on spatial relations</td></tr>
  <tr><td>MM-Vet (Spat)</td><td>Spatial awareness evaluation track</td></tr>
  <tr><td>What’s Up</td><td>Spatial relation in visual grounding</td></tr>
  <tr><td>CV-Bench</td><td>Visual-center spatial benchmark</td></tr>
  <tr><td>SRBench</td><td>The extrapolation of spatial benchmark</td></tr>
  <tr><td>MindCube</td><td>The extrapolation of spatial benchmark</td></tr>
  <tr><td>OmniSpatial</td><td>Comprehensive spatial reasoning dataset</td></tr>
  <tr><td>RealWorldQA</td><td>Comprehensive spatial reasoning dataset</td></tr>
  <tr>
    <td>ViewSpatial-Bench</td>
    <td>Multi-view spatial reasoning benchmark</td>
    <td><a href="https://huggingface.co/datasets/LLDDSS/Awesome_Spatial_VQA_Benchmarks_ViewSpatial-Bench">Link</a></td>
  </tr>
</table>


### 📖 Related Surveys
- Spatial reasoning in VLMs and embodied AI  
- 3D vision and scene understanding  
- Multimodal evaluation and benchmarking

---

## Citation
If you find this survey or repository useful for your research, please cite our paper:
```
@article{YourLastName2025SpatialSurvey,
  title   = {{Your Paper Title}},
  author  = {Your Name and Co-author 1 and Co-author 2},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2025},
  volume  = {},
  number  = {},
  pages   = {}
}
```