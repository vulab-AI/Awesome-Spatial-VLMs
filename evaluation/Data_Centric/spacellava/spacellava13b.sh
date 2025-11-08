#!/bin/bash
#SBATCH --job-name=spacellava13b           # 任务名称
#SBATCH -A aisc                      # 账户名
#SBATCH -p aisc                      # 分区（队列）名
#SBATCH --gres=gpu:1                 # 申请1块GPU
#SBATCH --mem=100G                   # 申请100GB内存
#SBATCH --cpus-per-task=10           # 每个任务使用10个CPU核心
#SBATCH --nodes=1                    # 1个节点
#SBATCH --ntasks=1                   # 1个任务
#SBATCH --output=13b.log    # 标准输出日志文件，%j是任务ID
#SBATCH --time=13-08:00:00
#SBATCH --error=spacellava13b.err      # 错误日志文件

# 加载环境，比如conda
module load GCCcore/13.3.0
source /home/yxy1421/.bashrc
conda activate spacellava

# 运行你的程序
python spacellava13b_eval.py >spacellava13b.log 2>&1



