# Copyright 2026 RL-bMAS Contributors
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

#测试黑板环境的脚本，用于直观的观察和黑板环境交互过程中黑板内容的变化
from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory
import torch
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq
import hydra

class BlackboardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self):
        obs, infos = self.envs.reset()
        self.memory.reset(batch_size = len(obs))
        self.pre_text_obs = obs
        full_text_obs = self.build_text_obs(obs, infos, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, infos)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, text_obs: List[Dict], infos: List[Dict], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        
        if init:
            for i in range(len(text_obs)):
                obs = BLACKBOARD_TEMPLATE_NO_HIS.format(
                    role_list=infos[i]['available_actions'],
                    question=infos[i]['current_task']
                )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                obs = BLACKBOARD_TEMPLATE.format(
                    role_list=infos[i]['available_actions'],
                    question=infos[i]['current_task'],
                    current_blackboard=str(text_obs[i]),
                    step_count=len(self.memory[i]),
                    #history_length=self.config.env.history_length,
                    #action_history=self.memory[i].get('action', []),
                    current_step=len(self.memory[i]) + 1,
                )
                postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['correct'])
                success['success_rate'].append(won_value)
                return

@hydra.main(config_path="test_config", config_name="test_blackboard_env", version_base=None)
def main(config):
    run_test(config)
    

def run_test(config):
    from agent_system.environments.env_package.blackboard import build_blackboard_envs, blackboard_projection
    train_file_path = config.data.train_files
    val_file_path = config.data.val_files
    #读取parquet文件获取训练和验证数据，存成list，list中每个元素是一个dict包含task和answer
    import pyarrow.parquet as pq
    train_data = pq.read_table(train_file_path).to_pydict()
    val_data = pq.read_table(val_file_path).to_pydict()
    train_data = [{'task': task, 'answer': answer} for task, answer in zip(train_data['task'], train_data['answer'])]
    val_data = [{'task': task, 'answer': answer} for task, answer in zip(val_data['task'], val_data['answer'])]
    dataset_config = {
        "train_tasks": train_data,
        "val_tasks": val_data,
    }

    _val_envs = build_blackboard_envs(seed=config.env.seed + 1000, config=dataset_config, env_num=config.data.val_batch_size, group_n=1, is_train=False)
    multiple_rollout_instance = multi_rollout(name_or_path=config.actor_rollout_ref.model.path, correct_pad_token=True, trust_remote_code=False)
    projection_f = partial(blackboard_projection)
    val_envs = BlackboardEnvironmentManager(_val_envs, projection_f, config)
    obs, infos = val_envs.reset()
    print("Initial Observation:")
    print(obs['text'][0])
    done = [False] * config.data.val_batch_size
    step_count = 0
    while not all(done) and step_count < 10:
        actions = []
        actions = multiple_rollout_instance.generate(obs['text'], max_new_tokens=1024, temperature=0.4, top_p=0.9)
        obs, rewards, done, infos = val_envs.step(actions)
        step_count += 1
    print("\nTest completed.")
    
class multi_rollout:
    '''
    用于并行调用大模型进行多次rollout
    '''
    def __init__(self, name_or_path, correct_pad_token=True, trust_remote_code=False):
        #加载本地模型
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        if correct_pad_token:
            self.set_pad_token_id(self.tokenizer)
        actor_model_config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2")
        if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
            actor_module_class = AutoModelForVision2Seq
        else:
            actor_module_class = AutoModelForCausalLM
        self.model = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=name_or_path,
                torch_dtype=torch.bfloat16,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
                device_map="auto",
            )
        self.model.eval()
        #self.model.to('cuda')
        # #使用多张gpu卡
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs")
        #     self.model = torch.nn.DataParallel(self.model)
        
    def generate(self, prompts: List[str], max_new_tokens=1024, temperature=1.0, top_p=1.0) -> List[str]:
        device = self.model.get_input_embeddings().weight.device
        self.tokenizer.padding_side = "left"
        prompt_texts = [self.tokenizer.apply_chat_template([{'role':'user','content':prompt}], tokenize=False) for prompt in prompts]
        #print('222222222222222222',prompt_texts)
        inputs = self.tokenizer(prompt_texts, return_tensors="pt", padding=True).to(device)
        #print(inputs)
        #inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        #input_ids = inputs["input_ids"]#.to(self.model.device)
        #attention_mask = inputs["attention_mask"]#.to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                #attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        sequences = sequences.to('cpu')
        # results = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        results = []
        for i, seq in enumerate(sequences):
            gen_ids = seq[input_lengths[i]:]
            results.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
        print(results)
        return results

    def set_pad_token_id(self, tokenizer):
        """Set pad_token_id to eos_token_id if it is None.

        Args:
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

        """
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)

if __name__ == "__main__":
    main()
