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
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [🚀 Awesome Papers](#-awesome-papers)
  - [Training-Free Prompting](#training-free-prompting)
    - [Textual Prompting Methods](#textual-prompting-methods)
    - [Visual Prompting Methods](#visual-prompting-methods)
    - [Hybrid Prompting](#hybrid-prompting)
  - [Model-Centric Enhancements](#model-centric-enhancements)
    - [Advanced Training Strategies](#advanced-training-strategies)
    - [Architectural Enhancements](#architectural-enhancements)
    - [Encoder-Level Improvements](#encoder-level-improvements)
  - [Explicit 2D Information Injection](#explicit-2d-information-injection)
    - [Object Region Guidance](#object-region-guidance)
    - [Explicit spatial relationship](#explicit-spatial-relationship)
  - [3D Information Enhancement](#3d-information-enhancement)
    - [Explicit 3D Geometric Representations](#explicit-3d-geometric-representations)
    - [Implicit 3D from Egocentric Views](#implicit-3d-from-egocentric-views)
    - [Scene-level information + Ego-centric](#scene-level-information--ego-centric)
  - [Data-Centric Spatial Enhancement](#data-centric-spatial-enhancement)
    - [Manifesting Spatial Relations in 2D Images](#manifesting-spatial-relations-in-2d-images)
    - [Manifesting Spatial Priors in 3D and Synthetic Worlds](#manifesting-spatial-priors-in-3d-and-synthetic-worlds)
- [📚 Datasets and Benchmarks](#-datasets-and-benchmarks)
  - [Spatially-Oriented Training Corpora](#spatially-oriented-training-corpora)
  - [Evaluation Benchmarks](#evaluation-benchmarks)
- [🏆 Spatial VLM Leaderboard \& Evaluation Toolkit](#-spatial-vlm-leaderboard--evaluation-toolkit)
  - [Leaderboard](#leaderboard)
  - [Evaluation Toolkit](#evaluation-toolkit)
  - [🏆 Main Leaderboard](#-main-leaderboard)
  - [🧑‍🔬 How to Evaluate Your Model](#-how-to-evaluate-your-model)
  - [📖 Related Surveys](#-related-surveys)
- [Citation](#citation)


## Overview
This repository uses the framework from our survey paper to systematically organize the field of Spatial Intelligence in VLMs.
- **The “What”: A Cognitive Hierarchy** 🧩  
  We define spatial intelligence as a 3-level hierarchy, and group tasks, datasets, and benchmarks by required capability:  
  **L1** *Perception* of intrinsic 3D attributes (e.g., size, orientation) &rarr; **L2** relational *Understanding* &rarr; **L3** *Extrapolation* (e.g., hidden-state inference, future prediction).
- **The “How”: A Taxonomy of Methods** 🚀  
  Methods are organized into five families, giving you a clear map of the current landscape. See details in [🚀 Awesome Papers](#🚀-awesome-papers).
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
#### Textual Prompting Methods

University of Toronto; Reasoning Paths with Reference Objects Elicit Quantitative Spatial Reasoning in Large Vision-Language Models (EMNLP2024). [[paper]]() [[code]](https://github.com/andrewliao11/Q-Spatial-Bench-code); 

University of California, Berkeley; Compositional Chain-of-Thought Prompting for Large Multimodal Models (CVPR2024). [[paper]]() [[code]](https://github.com/chancharikmitra/CCoT); 

Peking University; Enhancing the Spatial Awareness Capability of Multi-Modal Large Language Model (CoRR2023). [[paper]](); 

University of Oxford; SpatialPIN: Enhancing Spatial Reasoning Capabilities of Vision-Language Models through Prompting and Interacting 3D Priors (NeurIPS2024). [[paper]](); 

Tsinghua University; SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation (CoRR 2025). [[paper]]() [[code]](https://github.com/qizekun/SoFar); 

--------

#### Visual Prompting Methods

Nanjing University of Science and Technology; Fine-Grained Visual Prompting (NeurIPS2023). [[paper]]() [[code]](https://github.com/ylingfeng/FGVP); 

Microsoft Research, Redmond; Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V (arXiv2023). [[paper]]() [[code]](https://github.com/microsoft/SoM); 

Shanghai AI Lab; 3daxisprompt: Promoting the 3d grounding and reasoning in gpt-4o (Neurocomp’2025). [[paper]](); 

University of Washington; Coarse correspondences boost spatial-temporal reasoning in multimodal language model (CVPR2025). [[paper]](); 

Toyota Central R&D Labs; SpatialPrompting: Keyframe-driven Zero-Shot Spatial Reasoning with Off-the-Shelf Multimodal Large Language Models (arXiv2025). [[paper]](); 

UMass Amherst; Mindjourney: Test-time scaling with world models for spatial reasoning (arXiv2025). [[paper]]() [[code]](https://github.com/UMass-Embodied-AGI/MindJourney); 

--------

#### Hybrid Prompting

HKUST(Guangzhou); Seeground: See and ground for zero-shot open-vocabulary 3d visual grounding (CVPR 2025). [[paper]]() [[code]](https://github.com/iris0329/SeeGround); 

Tsinghua University; Scaffolding coordinates to promote vision-language coordination in large multi-modal models (COLING2025). [[paper]]() [[code]](https://github.com/THUNLP-MT/Scaffold); 

Microsoft Research; Mind’s Eye of LLMs: Visualization-of-Thought Elicits Spatial Reasoning in Large Language Models (NeurIPS2024). [[paper]]() [[code]](https://github.com/microsoft/visualization-of-thought/); 

Westlake University; Image-of-Thought Prompting for Visual Reasoning Refinement in Multimodal Large Language Models (arXiv2024). [[paper]](); 

University of Washington; Visual sketchpad: Sketching as a visual chain of thought for multimodal language models (NeurIPS2024). [[paper]]() [[code]](https://github.com/Yushi-Hu/VisualSketchpad); 

--------

### Model-Centric Enhancements
#### Advanced Training Strategies

Institute of Automation, Chinese Academy of Sciences; ROSS ( ICLR 2025). [[paper]]() [[code]](https://github.com/haochen-wang409/ross) [[checkpoint]](https://huggingface.co/HaochenWang/ross-qwen2-7b); 

Huawei Noah's Ark Lab; SpatialCoT (arXiv2025). [[paper]](); 

UT Austin; Cube-LLM ( ICLR 2025). [[paper]](); 

Microsoft Research; MVoT (ICML2025). [[paper]]() [[code]](https://github.com/chengzu-li/MVoT); 

Peking University; SpaceR (arXiv2025). [[paper]]() [[code]](https://github.com/OuyangKun10/SpaceR?tab=readme-ov-file) [[checkpoint]](https://huggingface.co/RUBBISHLIKE/SpaceR); 

Institute of Automation, Chinese Academy of Sciences.; ViLaSR (arXiv2025). [[paper]]() [[code]](https://github.com/AntResearchNLP/ViLaSR) [[checkpoint]](https://huggingface.co/inclusionAI/ViLaSR/tree/main); 

Inclusion AI, Ant Group; M2-Reasoning (arXiv2025). [[paper]]() [[code]](https://github.com/inclusionAI/M2-Reasoning) [[checkpoint]](https://huggingface.co/inclusionAI/M2-Reasoning); 

Stony Brook University; Svqa-r1: Reinforcing spatial reasoning in mllms via view-consistent reward optimization, (arXiv2025). [[paper]](); 

Johns Hopkins University; Spatialreasoner: Towards explicit and generalizable 3d spatial reasoning, (NeurIPS 2025). [[paper]]() [[code]](https://github.com/johnson111788/SpatialReasoner) [[checkpoint]](https://huggingface.co/collections/ccvl/spatialreasoner-68114caec81774edbf1781d3); 

Tsinghua University; Embodied-r: Collaborative framework for activating embodied spatial reasoning in foun2339 dation models via reinforcement learning (arXiv2025). [[paper]]() [[code]](https://github.com/EmbodiedCity/Embodied-R.code) [[checkpoint]](https://huggingface.co/EmbodiedCity/Embodied-R); 

Zhejiang University; Spatialladder: Progressive training for spatial reasoning in vision-language models (arXiv2025). [[paper]]() [[code]](https://github.com/zju-real/SpatialLadder) [[checkpoint]](https://huggingface.co/hongxingli/SpatialLadder-3B); 

Peking University; Spacer: Reinforcing mllms in video spatial reasoning (arXiv2025). [[paper]]() [[code]](https://github.com/OuyangKun10/SpaceR) [[checkpoint]](https://huggingface.co/RUBBISHLIKE/SpaceR); 

UIUC; Fine-Grained Preference Optimization Improves Spatial Reasoning in VLMs (NeurIPS 2025). [[paper]](); 

Northwestern University; Metaspatial: Reinforcing 3d spatial reasoning in vlms for the metaverse (arXiv2025). [[paper]]() [[code]](https://github.com/PzySeere/MetaSpatial); 

Stanford University; LOCALITY ALIGNMENT IMPROVES VISION-LANGUAGEMODELS (ICLR2025). [[paper]](); 

National University of Singapore 2ARC Lab,; What makes for good visual tokenizers for large language models (arXiv2023). [[paper]]() [[code]](https://github.com/TencentARC/GVT) [[checkpoint]](https://github.com/TencentARC/GVT/tree/master/gvt); 

University of Washington; Perception Tokens Enhance Visual Reasoning in Multimodal Language Models (CVPR 2025). [[paper]]() [[code]](https://github.com/mahtabbigverdi/Aurora-perception) [[checkpoint]](https://drive.google.com/file/d/1r7WYQWYA6VDpzfxPIHP1zEUgBYQmwNIj/view); 

--------

#### Architectural Enhancements

Digital Twin Institute, Eastern Institute of Technology, Ningbo, China; To Preserve or To Compress: An In-Depth Study of Connector Selection in Multimodal Large Language Models (EMNLP2024). [[paper]]() [[code]](https://github.com/EIT-NLP/Connector-Selection-for-MLLM); 

Kakao Brain; Honeybee: Locality enhanced projector for multimodal llm (CVPR2024 (Highlight)). [[paper]]() [[code]](https://github.com/khanrc/honeybee) [[checkpoint]](https://github.com/khanrc/honeybee); 

New York University; Cambrian-1: A fully open, vision-centric exploration of multimodal llm (NeurIPS 2024 (Oral)). [[paper]]() [[code]](https://github.com/cambrian-mllm/cambrian) [[checkpoint]](https://huggingface.co/collections/nyu-visionx/cambrian-1-models-666fa7116d5420e514b0f23c); 

City University of Hong Kong; Why is spatial reasoning hard for vlms? an attention mechanism perspective on focus areas (ICML2025). [[paper]]() [[code]](https://github.com/shiqichen17/AdaptVis) [[checkpoint]](https://github.com/shiqichen17/AdaptVis); 

unc; Contrastive region guidance: Improving grounding in vision-language mod2381 els without training (ECCV 2024 ). [[paper]]() [[code]](https://github.com/meetdavidwan/crg) [[checkpoint]](https://github.com/meetdavidwan/crg); 

--------

#### Encoder-Level Improvements

Zhejiang University; Spatialclip: Learning 3d-aware image representations from spatially discriminative language, (CVPR2025). [[paper]]() [[code]](https://github.com/SpatialVision/Spatial-CLIP); 

National University of Singapore; Introducing Visual Perception Token into Multimodal Large Language Model (arXiv2025). [[paper]]() [[code]](https://github.com/yu-rp/VisualPerceptionToken?tab=readme-ov-file) [[checkpoint]](https://huggingface.co/collections/rp-yu/vpt-models-67b6afdc8679a05a2876f07a); 

Stanford; Prismatic vlms: Investigating the design space of visually-conditioned language models (ICML 2024). [[paper]]() [[code]](https://github.com/TRI-ML/prismatic-vlms?tab=readme-ov-file#pretrained-models) [[checkpoint]](https://github.com/TRI-ML/prismatic-vlms); 

Huawei; Incorporating Visual Experts to Resolve the Information Loss in Multimodal Large Language Models (IJCAI2025). [[paper]](); 

Johns Hopkins University; Spatialllm (CVPR2025). [[paper]](); 

New York University; Cambrian-1 (NeurIPS 2024 (Oral)). [[paper]]() [[code]](https://github.com/cambrian-mllm/cambrian) [[checkpoint]](https://huggingface.co/collections/nyu-visionx/cambrian-1-models-666fa7116d5420e514b0f23c); 

University of Illinois Urbana-Champaign; ARGUS (CVPR2025). [[paper]](); 

NVIDIA; Eagle 2 (ICLR 2025). [[paper]]() [[code]](https://github.com/NVlabs/EAGLE?tab=readme-ov-file) [[checkpoint]](https://huggingface.co/nvidia/Eagle2.5-8B); 

--------


### Explicit 2D Information Injection
#### Object Region Guidance

Nvidia; RegionGPT  (CVPR2024). [[paper]](); 

The University of Hong Kong; GPT4RoI (ECCV 2024 Workshops). [[paper]]() [[code]](https://github.com/jshilong/GPT4RoI?tab=readme-ov-file) [[checkpoint]](https://huggingface.co/shilongz/GPT4RoI-7B-delta-V0); 

Tsinghua University; PVIT (CoRR 2023). [[paper]]() [[code]](https://github.com/PVIT-official/PVIT?tab=readme-ov-file#pvit-weights) [[checkpoint]](https://huggingface.co/PVIT/pvit); 

Georgia Tech; VCoder (CVPR2024). [[paper]]() [[code]](https://github.com/SHI-Labs/VCoder) [[checkpoint]](https://huggingface.co/models?search=vcoder); 

University of Illinois Urbana-Champaign; Argus: Vision-centric reasoning with grounded chain-of-thought (CVPR2025). [[paper]](); 

UMass Amhers; CoVLM (ICLR 2024). [[paper]]() [[code]](https://github.com/UMass-Embodied-AGI/CoVLM?tab=readme-ov-file) [[checkpoint]](https://github.com/UMass-Embodied-AGI/CoVLM?tab=readme-ov-file); 

Tsinghua University; PEVL (EMNLP 2022 ). [[paper]]() [[code]](https://github.com/thunlp/PEVL) [[checkpoint]](https://github.com/thunlp/PEVL); 

Meta; Learning to Localize Objects Improves Spatial Reasoning in Visual-LLMs (CVPR2024). [[paper]](); 

International Digital Economy Academy; Lyrics (arXiv2025). [[paper]](); 

--------


#### Explicit spatial relationship

Tel-Aviv University; SGVL (ACL2023). [[paper]]() [[code]](https://github.com/AlonMendelson/SGVL) [[checkpoint]](https://drive.google.com/file/d/13jzpcLgGalO3hkiqVwziNAlCEZD90ENN/view); 

meta; Object-centric Binding in Contrastive Language-Image Pretraining (arXiv2025). [[paper]](); 

South China University of Technology; Seeing Beyond the Scene: Enhancing Vision-Language Models with Interactional Reasoning (arXiv2025). [[paper]](); 

Tsinghua University; LLaVA-SG (arXiv2025). [[paper]](); 

--------

### 3D Information Enhancement
#### Explicit 3D Geometric Representations

Beijing Digital Native Digital City Research Center; Spatial 3d-llm: Progressive spatial awareness for advanced 3d vision-language understanding (arXiv2025). [[paper]](); 

Nanyang Technological University; Segpoint: Segment any point cloud via large language model (ECCV2024). [[paper]](); 

Fudan University; Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning (CVPR2024). [[paper]]() [[code]](https://github.com/Open3DA/LL3DA); 

Kyoto Universit; Scanqa: 3d question answering for spatial scene understanding (CVPR2022). [[paper]]() [[code]](https://github.com/ATR-DBI/ScanQA); 

UCLA; 3d-llm: Injecting the 3d world into large language models (NeurIPS2023). [[paper]]() [[code]](https://github.com/UMass-Embodied-AGI/3D-LLM); 

The University of Adelaide; 3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer (CVPR2025). [[paper]]() [[code]](https://github.com/djiajunustc/3D-LLaVA?tab=readme-ov-file) [[checkpoint]](https://huggingface.co/djiajunustc/3D-LLaVA-7B-LoRA); 

UIUC; Situational Awareness Matters in 3D Vision Language Reasoning (CVPR 2024). [[paper]]() [[code]](https://github.com/YunzeMan/Situation3D); 

South China University of Technology; Lscenellm: Enhancing large 3d scene understanding using adaptive visual preferences (CVPR 2025). [[paper]]() [[code]](https://github.com/Hoyyyaard/LSceneLLM) [[checkpoint]](https://huggingface.co/Hoyard/LSceneLLM); 

Georgia Tech; Vcoder: Versatile vision encoders for  multimodal large language model (CVPR2024). [[paper]]() [[code]](https://github.com/SHI-Labs/VCoder) [[checkpoint]](https://huggingface.co/models?search=vcoder); 

Shanghai Jiao Tong University; SpatialBot: Precise Spatial Understanding with Vision Language Models (ICRA2025). [[paper]]() [[code]](https://github.com/BAAI-DCAI/SpatialBot) [[checkpoint]](https://huggingface.co/RussRobin/SpatialBot-3B); 

UC San Diego; SpatialRGPT: Grounded Spatial Reasoning in Vision-Language Models (NeurIPS2024). [[paper]]() [[code]](https://github.com/AnjieCheng/SpatialRGPT) [[checkpoint]](https://huggingface.co/collections/a8cheng/spatialrgpt-grounded-spatial-reasoning-in-vlms-66fef10465966adc81819723); 

Universit ́e de Moncton; SmolRGPT: Efficient Spatial Reasoning for Warehouse Environments with 600M Parameters (ICCVW'25). [[paper]]() [[code]](https://github.com/abtraore/SmolRGPT) [[checkpoint]](https://huggingface.co/collections/Abdrah/smolrgpt-checkpoints-6893bad56127440ef250486e); 

Beihang University; RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics (NeurIPS2025). [[paper]]() [[code]](https://github.com/Zhoues/RoboRefer) [[checkpoint]](https://huggingface.co/collections/Zhoues/roborefer-and-refspatial-6857c97848fab02271310b89); 

Westlake University; SSR: Enhancing Depth Perception in Vision-Language Models via Rationale-Guided Spatial Reasoning (NeurIPS 2025). [[paper]]() [[code]](https://github.com/yliu-cs/SSR) [[checkpoint]](https://huggingface.co/collections/yliu-cs/ssr-682d44496b64e4edd94092bb); 

Zhejiang University; SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models (NeurIPS2025). [[paper]]() [[code]](https://github.com/cpystan/SD-VLM) [[checkpoint]](https://huggingface.co/cpystan/SD-VLM-7B); 

--------

#### Implicit 3D from Egocentric Views

UT Austin; VLM-3R: Vision-Language Models Augmented with Instruction-Aligned 3D Reconstruction (arXiv2025). [[paper]]() [[code]](https://github.com/VITA-Group/VLM-3R); 

Tsinghua University; Spatial-MLLM: Boosting MLLM Capabilities in Visual-based Spatial Intelligence (arXiv2025). [[paper]]() [[code]](https://github.com/diankun-wu/Spatial-MLLM); 

Georgia Institute of Technology; SplatTalk: 3D VQA with Gaussian Splatting (ICCV 2025). [[paper]]() [[code]](https://splat-talk.github.io/); 

Guangdong Polytechnic Normal Universit; I Know About “Up”! Enhancing Spatial Reasoning in Visual Language Models Through 3D Reconstruction (arXiv2024). [[paper]](); 

--------

#### Scene-level information + Ego-centric

The University of Hong Kong; LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities (ICCV 2025). [[paper]]() [[code]](https://github.com/ZCMax/LLaVA-3D) [[checkpoint]](https://huggingface.co/ChaimZhu/LLaVA-3D-7B); 

UCLA; 3D Concept Learning and Reasoning from Multi-View Images (CVPR2023). [[paper]]() [[code]](https://github.com/evelinehong/3D-CLR-Official); 

The University of Hong Kong; Gpt4scene: Understand 3d scenes from videos with vision-language models (arXiv2025). [[paper]]() [[code]](https://github.com/Qi-Zhangyang/GPT4Scene-and-VLN-R1) [[checkpoint]](https://huggingface.co/alexzyqi/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512); 

Brown University; Scene-LLM: Extending Language Model for 3D Visual Understanding and Reasoning (WACV2025). [[paper]](); 

Zhejiang University; Chat-scene: Bridging 3d scene and large language models with object identifiers (NeurIPS2024). [[paper]]() [[code]](https://github.com/ZzZZCHS/Chat-Scene); 

University of Illinois Chicago; Robin3D: Improving 3D Large Language Model via Robust Instruction Tuning (ICCV 2025). [[paper]]() [[code]](https://github.com/WeitaiKang/Robin3D?tab=readme-ov-file) [[checkpoint]](https://drive.google.com/drive/folders/14Si8bdWI3N5NEeVDLhmAlxilWPl0f_Wp?usp=sharing); 

Zhejiang University; Inst3D-LMM: Instance-Aware 3D Scene Understanding with Multi-modal Instruction Tuning (CVPR2025). [[paper]]() [[code]](https://github.com/hanxunyu/Inst3D-LMM); 

Sun Yat-sen University,; DSPNet: Dual-vision Scene Perception for Robust 3D Question Answering (CVPR2025). [[paper]]() [[code]](https://github.com/LZ-CH/DSPNet) [[checkpoint]](https://github.com/LZ-CH/DSPNet); 

Beijing Institute for General Artificial Intelligence (BIGAI); An Embodied Generalist Agent in 3D World (ICML2024). [[paper]]() [[code]](https://github.com/embodied-generalist/embodied-generalist) [[checkpoint]](https://huggingface.co/datasets/huangjy-pku/LEO_data/tree/main); 

The University of Hong Kong; ScanReason: Empowering 3D Visual Grounding with Reasoning Capabilities (ECCV2024). [[paper]]() [[code]](https://github.com/ZCMax/ScanReason); 

Apple; MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs (ICCV 2025). [[paper]]() [[code]](https://github.com/apple/ml-cubifyanything); 

--------

### Data-Centric Spatial Enhancement

#### Manifesting Spatial Relations in 2D Images

Tsinghua University; Position-Enhanced Visual Instruction Tuning for Multimodal Large Language Models (arXiv23). [[paper]]() [[code]](https://github.com/PVIT-official/PVIT) [[checkpoint]](https://huggingface.co/PVIT/pvit); 

University of Waterloo; SpaRE (arXiv25). [[paper]](); 

Shanghai AI Laboratory; The all-seeing project v2: Towards general relation comprehension of the open world (ECCV24). [[paper]]() [[code]](https://github.com/OpenGVLab/all-seeing?tab=readme-ov-file) [[checkpoint]](https://huggingface.co/OpenGVLab/ASMv2); 

Microsoft Research; Kosmos-2  (ICLR24). [[paper]]() [[code]](https://github.com/microsoft/unilm/tree/master/kosmos-2) [[checkpoint]](https://huggingface.co/microsoft/kosmos-2-patch14-224); 

Tsinghua University; Pseudo-Q (CVPR22). [[paper]]() [[code]](https://github.com/LeapLabTHU/Pseudo-Q?tab=readme-ov-file); 

--------

#### Manifesting Spatial Priors in 3D and Synthetic Worlds

Google DeepMind; SpatialVLM (CVPR24). [[paper]]() [[code]](https://spatial-vlm.github.io/#community-implementation) [[checkpoint]](https://github.com/remyxai/VQASynth?tab=readme-ov-file#models-trained-using-vqasynth-); 

City University of Hong Kong; LLaVA-SpaceSGG (WACV25). [[paper]]() [[code]](https://github.com/Endlinc/LLaVA-SpaceSGG?tab=readme-ov-file) [[checkpoint]](https://huggingface.co/wumengyangok/LLaVA-SpaceSGG/tree/main); 

NVIDIA; RoboSpatial (CVPR25). [[paper]]() [[code]](https://github.com/NVlabs/RoboSpatial); 

Michigan State University & UC Davis; SPARTUN3D (ICLR25). [[paper]](); 

BIGAI; MSQA (NeurIPS24). [[paper]]() [[code]](https://github.com/MSR3D/MSR3D); 

Meta FAIR; MultiSPA (arXiv25). [[paper]]() [[code]](https://github.com/facebookresearch/Multi-SpatialMLLM?tab=readme-ov-file#-model-training); 

Massachusetts Institute of Technology; Sparkle (arXiv25). [[paper]](); 

Zhejiang University ; Orient Anything (ICML25). [[paper]]() [[code]](https://github.com/SpatialVision/Orient-Anything?tab=readme-ov-file) [[checkpoint]](https://huggingface.co/Viglong/Orient-Anything/blob/main/croplargeEX2/dino_weight.pt); 


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