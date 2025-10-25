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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import json

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import  GcotRewardManager     # CustomRewardManager,
from .config import PPOConfig
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role

import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        config.deep_post_init()
        print(json.dumps(config.to_dict(), indent=2))

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

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
            Role.RefPolicy: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        reward_fn = GcotRewardManager(tokenizer=tokenizer, compute_score=config.worker.reward.compute_score)
        val_reward_fn = GcotRewardManager(tokenizer=tokenizer, compute_score=config.worker.reward.compute_score)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config = OmegaConf.to_object(ppo_config)
    ppo_config.data.system_prompt = """\n### Guidance:\nYou are a spatial reasoning assistant with access to two powerful visualization tools.\nYour task is to break down complex spatial problems and iteratively refine your solution through visualization feedback.\n\nAvailable tools:\nYou can use below two tools to visualize, and you should wait the visualization feedback for refine your reasoning.\n\n1. **Object Mapper**\n- Purpose: Identifies and maps key items in the space\n- Input format: JSON\n```json\n[{\n    "bbox_2d": [x1, y1, x2, y2],\n    "label": "object name/description"\n}]\n```\n- Output: Generates bounding boxes for visual inspection\n\n2. **Path Tracer**\n- Purpose: Plots movement or connections between points\n- Input format: JSON\n```json\n[{\n    "start_point_2d": [x1, y1],\n    "end_point_2d": [x2, y2],\n    "label": "trace_description"\n}]\n```\n- Output: Generates visual paths for verification\n\n### Required Output Format:\nFor each reasoning step, you must structure your response as follows:\n<think> [Your detailed reasoning process] </think> Action: [Object Mapper/Path Tracer]\n```json\n[JSON format coordinates]\n```\n\nAfter your reasoning and iteratively refine your solution through visualization feedback, you should arrive at a final answer and structure your response as follows:\n<think> [Your detailed reasoning process] </think> Action: Answer\n<answer> [Your final answer] </answer>\n\n### Solution Process:\n1. Initial Analysis\n   - Break down the spatial problem\n   - Plan your approach\n\n2. Iterative Reasoning\n   For each step:\n   - Choose appropriate tool\n   - Provide coordinates in JSON format\n   - Observe the visualization output\n   - Reflect on the results:\n     * Is the placement/path accurate?\n     * Does it align with your reasoning?\n     * What adjustments are needed?\n   - Refine coordinates if necessary\n\n3. Validation\n   - Review complete solution\n   - Ensure all steps logically connect\n   - Verify final answer matches visual evidence\n\n4. Backtrack and Adjust\n   - If errors are found, backtrack to previous steps\n   - Modify actions or decisions as needed\n   - Re-execute subsequent steps to validate corrections\n\n5. Summary and Ouput\n   - Summarize the entire process and give final answer"""


    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))


if __name__ == "__main__":
    main()
