# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import glob

from torch.utils.tensorboard import SummaryWriter

from ..py_functional import convert_dict_to_str, flatten_dict, is_package_available, unflatten_dict
from .gen_logger import AggregateGenerationsLogger


if is_package_available("mlflow"):
    import mlflow  # type: ignore


if is_package_available("wandb"):
    import wandb  # type: ignore


if is_package_available("swanlab"):
    import swanlab  # type: ignore


class Logger(ABC):
    @abstractmethod
    def __init__(self, config: Dict[str, Any]) -> None: ...

    @abstractmethod
    def log(self, data: Dict[str, Any], step: int) -> None: ...

    def finish(self) -> None:
        pass


class ConsoleLogger(Logger):
    def __init__(self, config: Dict[str, Any]) -> None:
        print("Config\n" + convert_dict_to_str(config))

    def log(self, data: Dict[str, Any], step: int) -> None:
        print(f"Step {step}\n" + convert_dict_to_str(unflatten_dict(data)))


class MlflowLogger(Logger):
    def __init__(self, config: Dict[str, Any]) -> None:
        mlflow.start_run(run_name=config["trainer"]["experiment_name"])
        mlflow.log_params(flatten_dict(config))

    def log(self, data: Dict[str, Any], step: int) -> None:
        mlflow.log_metrics(metrics=data, step=step)


# class TensorBoardLogger(Logger):
#     def __init__(self, config: Dict[str, Any]) -> None:
#         tensorboard_dir = os.getenv("TENSORBOARD_DIR", "tensorboard_log")
#         os.makedirs(tensorboard_dir, exist_ok=True)
#         print(f"Saving tensorboard log to {tensorboard_dir}.")
#         self.writer = SummaryWriter(tensorboard_dir)
#         self.writer.add_hparams(flatten_dict(config), metric_dict={'initialization': 0.0})

#     def log(self, data: Dict[str, Any], step: int) -> None:
#         for key, value in data.items():
#             self.writer.add_scalar(key, value, step)

#     def finish(self):
#         self.writer.close()

class TensorBoardLogger(Logger):
    def __init__(self, config: Dict[str, Any], exp_name: str = None) -> None:
        # 创建带时间戳的目录名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = exp_name or 'experiment'
        self.log_dir = os.path.join(
            os.getenv("TENSORBOARD_DIR", "path/to/tensorboard_log"), # ./tensorboard_log
            f"{exp_name}_{timestamp}"
        )
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"Saving tensorboard log to {self.log_dir}")
        
        # 初始化writer
        self.writer = SummaryWriter(self.log_dir)
        self.writer.add_hparams(flatten_dict(config), metric_dict={'initialization': 0.0})
        
        # 保存配置到文本文件
        self._save_config(config)

    def _save_config(self, config: Dict):
        """保存配置到文本文件"""
        config_path = os.path.join(self.log_dir, 'config.txt')
        with open(config_path, 'w') as f:
            f.write("Configuration:\n")
            f.write("="*50 + "\n")
            for k, v in flatten_dict(config).items():
                f.write(f"{k}: {v}\n")

    def log(self, data: Dict[str, Any], step: int) -> None:
        """记录数据并保存到文本文件"""
        # 写入TensorBoard
        for key, value in data.items():
            self.writer.add_scalar(key, value, step)
        
        # 同时写入文本文件
        metrics_path = os.path.join(self.log_dir, 'metrics.txt')
        with open(metrics_path, 'a') as f:
            f.write(f"Step {step}:\n")
            for key, value in data.items():
                f.write(f"  {key}: {value}\n")

    def finish(self):
        self.writer.close()
        print(f"\nLogging finished. Files saved in {self.log_dir}:")
        for file in glob.glob(os.path.join(self.log_dir, '*')):
            print(f"- {os.path.basename(file)}")


class WandbLogger(Logger):
    def __init__(self, config: Dict[str, Any]) -> None:
        wandb.init(
            project=config["trainer"]["project_name"],
            name=config["trainer"]["experiment_name"],
            mode="offline",                         # modified by junfei wu
            dir='./wandb_logs',                     # modified by junfei wu
            config=config,
        )

    def log(self, data: Dict[str, Any], step: int) -> None:
        wandb.log(data=data, step=step)

    def finish(self) -> None:
        wandb.finish()


class SwanlabLogger(Logger):
    def __init__(self, config: Dict[str, Any]) -> None:
        swanlab_key = os.getenv("SWANLAB_API_KEY")
        swanlab_dir = os.getenv("SWANLAB_DIR", "swanlab_log")
        swanlab_mode = os.getenv("SWANLAB_MODE", "cloud")
        if swanlab_key:
            swanlab.login(swanlab_key)

        swanlab.init(
            project=config["trainer"]["project_name"],
            experiment_name=config["trainer"]["experiment_name"],
            config={"UPPERFRAMEWORK": "EasyR1", "FRAMEWORK": "veRL", **config},
            logdir=swanlab_dir,
            mode=swanlab_mode,
        )

    def log(self, data: Dict[str, Any], step: int) -> None:
        swanlab.log(data=data, step=step)

    def finish(self) -> None:
        swanlab.finish()


LOGGERS = {
    "wandb": WandbLogger,
    "mlflow": MlflowLogger,
    "tensorboard": TensorBoardLogger,
    "console": ConsoleLogger,
    "swanlab": SwanlabLogger,
}


class Tracker:
    def __init__(self, loggers: Union[str, List[str]] = "console", config: Optional[Dict[str, Any]] = None):
        if isinstance(loggers, str):
            loggers = [loggers]

        self.loggers: List[Logger] = []
        for logger in loggers:
            if logger not in LOGGERS:
                raise ValueError(f"{logger} is not supported.")

            self.loggers.append(LOGGERS[logger](config))

        self.gen_logger = AggregateGenerationsLogger(loggers)

    def log(self, data: Dict[str, Any], step: int) -> None:
        for logger in self.loggers:
            logger.log(data=data, step=step)

    def log_generation(self, samples: List[Tuple[str, str, float]], step: int) -> None:
        self.gen_logger.log(samples, step)

    def __del__(self):
        for logger in self.loggers:
            logger.finish()

if __name__ == "__main__":
    test_config = {
        "model": {
            "name": "gpt2",
            "params": {
                "hidden_size": 768,
                "num_layers": 12
            }
        },
        "training": {
            "batch_size": 32,
            "lr": 0.001
        }
    }

    # 创建logger实例
    logger = TensorBoardLogger(test_config, exp_name="reward_test")
    
    print(f"\nStarting logging experiment...")
    print(f"Log directory: {logger.log_dir}")
    
    # 记录测试数据
    for i in range(10):
        metrics = {
            "acc_reward": i * 0.1,
            "format_reward": i * 0.2
        }
        logger.log(metrics, step=i)
        print(f"Logged step {i}: {metrics}")
    
    # 完成记录
    logger.finish()
    
    # 显示如何查看结果
    print("\nTo view the results:")
    print(f"1. Run TensorBoard:")
    print(f"   tensorboard --logdir={logger.log_dir}")
    print(f"\n2. View text logs directly:")
    print(f"   - Configuration: {os.path.join(logger.log_dir, 'config.txt')}")
    print(f"   - Metrics: {os.path.join(logger.log_dir, 'metrics.txt')}")
