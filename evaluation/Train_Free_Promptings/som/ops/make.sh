# #!/usr/bin/env bash
# # ------------------------------------------------------------------------------------------------
# # Deformable DETR
# # Copyright (c) 2020 SenseTime. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# # ------------------------------------------------------------------------------------------------
# # Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# # ------------------------------------------------------------------------------------------------

# # Copyright (c) Facebook, Inc. and its affiliates.
# # Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR
# # Modified by Richard Abrich from https://github.com/OpenAdaptAI/OpenAdapt

# # from https://github.com/pytorch/extension-cpp/issues/71#issuecomment-1778326052
# CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
# if [[ ${CUDA_VERSION} == 9.0* ]]; then
#     export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;7.0+PTX"
# elif [[ ${CUDA_VERSION} == 9.2* ]]; then
#     export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0+PTX"
# elif [[ ${CUDA_VERSION} == 10.* ]]; then
#     export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5+PTX"
# elif [[ ${CUDA_VERSION} == 11.0* ]]; then
#     export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0+PTX"
# elif [[ ${CUDA_VERSION} == 11.* ]]; then
#     export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
# elif [[ ${CUDA_VERSION} == 12.* ]]; then
#     export TORCH_CUDA_ARCH_LIST="5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX"
# else
#     echo "unsupported cuda version."
#     exit 1
# fi

# python -m pip install git+https://github.com/facebookresearch/detectron2.git

# python setup.py build install
#!/usr/bin/env bash

# 自动找 nvcc，而不是写死路径
NVCC_PATH=$(which nvcc)
if [ -z "$NVCC_PATH" ]; then
    echo "❌ nvcc not found in PATH. Please module load CUDA first."
    exit 1
fi

CUDA_VERSION=$($NVCC_PATH --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
echo "Using CUDA version: ${CUDA_VERSION}"

# 根据 CUDA 版本设置编译架构
if [[ ${CUDA_VERSION} == 9.0* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;7.0+PTX"
elif [[ ${CUDA_VERSION} == 9.2* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0+PTX"
elif [[ ${CUDA_VERSION} == 10.* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5+PTX"
elif [[ ${CUDA_VERSION} == 11.0* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0+PTX"
elif [[ ${CUDA_VERSION} == 11.* ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
elif [[ ${CUDA_VERSION} == 12.* ]]; then
    export TORCH_CUDA_ARCH_LIST="5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX"
else
    echo "❌ unsupported cuda version: ${CUDA_VERSION}"
    exit 1
fi

# 如果你已经装了 detectron2，可以注释掉这行
# python -m pip install git+https://github.com/facebookresearch/detectron2.git

# 编译 ops
python setup.py build install
