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
Generate responses given a dataset of prompts
"""
import numpy as np

import json

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup, RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import CustomRewardManager
from .config import PPOConfig
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from ..protocol import DataProto
from datasets import load_dataset



import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

# from verl.utils.model import compute_position_id_with_mask


@ray.remote(num_cpus=1)
class Runner:
    """A runner for Generation."""

    def run(self, config: PPOConfig):
        # print config
        config.deep_post_init()
        print(json.dumps(config.to_dict(), indent=2))

        data_path = config.data.val_files
        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"
        dataset = load_dataset(data_path, split=data_split)
        chat_lst = dataset[config.data.prompt_key]

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(FSDPWorker), config=config.worker, role='actor_rollout')
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()

        total_samples = len(dataset)
        # real_batch_size = data.batch['input_ids'].shape[0]
        config_batch_size = config.data.val_batch_size
        dp_size = wg.world_size // config.worker.rollout.tensor_parallel_size
        num_batch = (total_samples // config_batch_size) + 1
        output_lst = [[] for _ in range(1)]
        import pdb
        pdb.set_trace()

        for batch_idx in range(num_batch):
            print(f'[{batch_idx+1}/{num_batch}] Start to process.')
            batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
            inputs = tokenizer.apply_chat_template(batch_chat_lst,
                                                    add_generation_prompt=True,
                                                    padding=True,
                                                    truncation=True,
                                                    max_length=config.data.max_prompt_length,
                                                    return_tensors='pt',
                                                    return_dict=True,
                                                    tokenize=True)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask,}
            data = DataProto.from_dict(batch_dict)
            real_batch_size = data.batch['input_ids'].shape[0]
            pdb.set_trace()
        
            if real_batch_size % dp_size != 0:
                dummy_data_size = dp_size - real_batch_size % dp_size
                dummy_data = data[:dummy_data_size]
                data = DataProto.concat([data, dummy_data])
                print(
                    f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
                )
            
            batch_size = data.batch['input_ids'].shape[0]
            assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'
            print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
            pdb.set_trace()

            # START TO GENERATE FOR n_samples TIMES
            n_samples = 1
            for i in range(n_samples):
                output = wg.generate_sequences(data)
                # remove dummy data
                output = output[:real_batch_size]
                output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                                    skip_special_tokens=False)

                # remove the padding
                pad_token = tokenizer.pad_token
                output_text_unpad = []
                for text in output_text:
                    output_text_unpad.append(text.replace(pad_token, ''))

                output_lst[i].extend(output_text_unpad)

        # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
        output_lst = np.array(output_lst, dtype=object)
        output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

        # # add to the data frame
        # dataset[f'responses'] = output_lst

        return output_text


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    generation_config = OmegaConf.merge(default_config, cli_args)
    generation_config = OmegaConf.to_object(generation_config)

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    runner = Runner.remote()
    ray.get(runner.run.remote(generation_config))


if __name__ == '__main__':
    main()
