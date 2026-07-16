# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )
        # #将prompt的token大于32768的部分进行截断
        # if len(self.tokenizer.encode(prompt_with_chat_template, add_special_tokens=False)) > 32768:
        #     print(f"Warning: prompt length {len(self.tokenizer.encode(prompt_with_chat_template, add_special_tokens=False))} exceeds max_prompt_length {32768}, truncating...")
        #     prompt_with_chat_template = self.tokenizer.truncate_chat_prompt(
        #         prompt_with_chat_template,
        #         max_length=32768,
        #         add_generation_prompt=True,
        #     )
        #     prompt_with_chat_template = self.tokenizer.apply_chat_template(
        #         prompt_with_chat_template,
        #         add_generation_prompt=True,
        #         tokenize=False
        #     )

        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        # --- 爷爷请添加这段“强制截断”代码 START ---
        # 这是一个双保险。如果 verl_F 没有截断，我们在这里手动切一刀。
        # 注意：input_ids 通常是一个 list of list，所以我们要操作 input_ids[0]
        
        # 1. 打印一下真实的长度，帮您确认是不是黑板太长了
        real_token_len = len(input_ids[0])
        #print(f"Step debug: 当前 Prompt Token 长度为 {real_token_len}")

        # 2. 如果超长，强制截断
        # 这里的 28000 是您设置的 max_prompt_length
        limit = self.config.data.max_prompt_length 
        if real_token_len > limit:
            print(f"⚠️ 警告: 长度 {real_token_len} 超过限制 {limit}，正在强制截断...")
            
            # 【策略选择】
            # 策略 A (推荐): 保留最后面的内容（最新的指令和黑板末尾）。
            # 风险：可能会把最开头的 System Prompt (角色定义) 给切掉。
            input_ids[0] = input_ids[0][-limit:]
            attention_mask[0] = attention_mask[0][-limit:]
            
            # 策略 B (如果您的角色定义非常重要): 
            # 建议您在环境(Environment)里生成 blackboard 字符串时，
            # 就先检查一下字符串长度，不要让它太长，因为在这里切断 Token 很容易把关键指令切坏。
        # --- 爷爷请添加这段“强制截断”代码 END ---

        if is_multi_modal:

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)
        
        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': self.tokenizer.encode(raw_prompt, add_special_tokens=False),
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            episode_input_token_num: np.ndarray,
            episode_output_token_num: np.ndarray,
            episode_action_history_dict: Dict[str, int],
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            episode_input_token_num (np.ndarray): Total input tokens for each environment
            episode_output_token_num (np.ndarray): Total output tokens for each environment

        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        episode_rewards_mean = np.mean(episode_rewards)
        episode_rewards_min = np.min(episode_rewards)
        episode_rewards_max = np.max(episode_rewards)

        episode_lengths_mean = np.mean(episode_lengths)
        episode_lengths_min = np.min(episode_lengths)
        episode_lengths_max = np.max(episode_lengths)

        input_token_num = np.sum(episode_input_token_num)
        output_token_num = np.sum(episode_output_token_num)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    data['episode_lengths_mean'] = episode_lengths_mean
                    data['episode_lengths_min'] = episode_lengths_min
                    data['episode_lengths_max'] = episode_lengths_max
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value
                    #token num
                    data['total_input_token_num'] = input_token_num
                    data['total_output_token_num'] = output_token_num
                    #history
                    data['episode_action_history_dict'] = episode_action_history_dict
                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """
        import os, json, time
        from numpy import array
        SAVE_DIR = os.getenv("VERL_TRAJECTORY_SAVE_DIR", os.path.join(os.getcwd(), "trajectories"))
        os.makedirs(SAVE_DIR, exist_ok=True)

        def _save_one_trajectory(id, traj_uid, question, answer, history_steps, infos, blackboard):
            from numpy import array
            record = {
                "traj_uid": traj_uid,
                "question": question,
                "answer": answer,
                "steps": history_steps,   # 每一步的 obs / action / reward 等
                "correct": eval(infos[-1])['correct'],
                "blackboard": blackboard,
            }
            path = os.path.join(SAVE_DIR, f"{id}-{traj_uid}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2, default=str)

        # # Initial observations from the environment
        # obs, infos = envs.reset()
        # print('start rollout loop...')
        # # Initialize trajectory collection
        # lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        # if len(gen_batch.batch) != lenght_obs and self.config.env.rollout.n > 0:
        #     gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        # assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        length_obs = envs.get_num_envs()
        if len(gen_batch.batch) != length_obs and self.config.env.rollout.n > 0:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        assert len(gen_batch.batch) == length_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {length_obs}"

        batch_size = len(gen_batch.batch['input_ids'])
        batch_output = None
        question_list = []
        print('batch_size',batch_size)
        for item in range(batch_size):
            raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]######################################################################gen_batch有answer好像就够了，不用继续进入batch
            answer = gen_batch.non_tensor_batch['answer'][item]
            answer_type = gen_batch.non_tensor_batch['answer_type'][item]
            decision_prompt = gen_batch.non_tensor_batch['decision_prompt'][item]
            tmp_question = {'question': raw_prompt[0]['content'], 'answer': answer,'decision_prompt':decision_prompt, 'answer_type':answer_type}
            question_list.append(tmp_question)
        obs, infos = envs.reset(questions=question_list)
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.int32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        episode_input_token_num = np.zeros(batch_size, dtype=np.int32)
        episode_output_token_num = np.zeros(batch_size, dtype=np.int32)
        episode_action_count_dict_sum = {}#########################################
        episode_action_history_dict = {}
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            episode_action_count_dict_sum = {}#########################################
            active_masks = np.logical_not(is_done)
            # print('gen_batch:', gen_batch[0])
            # print('obs:', obs['text'][0])
            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info
            # print('batch_input:', batch_input[0])
            print('step:', _step)
            batch_output = actor_rollout_wg.generate_sequences(batch_input)
            print('finish generation')
            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            #print(text_actions)
            next_obs, rewards, dones, infos = envs.step(text_actions)

            
            #统计agent消耗的token数量
            for i in range(batch_size):
                episode_input_token_num[i] = infos[i]['input_token_num']
                episode_output_token_num[i] = infos[i]['output_token_num']
                #统计每个agent的action_count_dict
                action_count_dict = infos[i]['action_count_dict']
                for key, value in action_count_dict.items():
                    if key not in episode_action_count_dict_sum:
                        episode_action_count_dict_sum[key] = 0
                    episode_action_count_dict_sum[key] += value
            print('episode_action_count_dict_sum:', episode_action_count_dict_sum)
            #统计envs中action history分布
            #只在最后一个step统计
            # if _step == self.config.env.max_steps - 1:
            #     for i in range(batch_size):
            #         action_history = infos[i]['action_history']
            #         action_history_str = ','.join(action_history)
            #         if action_history_str not in episode_action_history_dict:
            #             episode_action_history_dict[action_history_str] = 0
            #         episode_action_history_dict[action_history_str] += 1
            #     print('episode_action_history_dict:', episode_action_history_dict)
            # #batch.non_tensor_batch['episode_action_count_dict_sum'] = episode_action_count_dict_sum

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            # Create reward tensor, only assign rewards for active environments
            episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)

            #只在最后一个step统计
            if is_done.all():
                print(f"All environments done at step {_step}.")
                for i in range(batch_size):
                    action_history = infos[i]['action_history']
                    action_history_str = ','.join(action_history)
                    if action_history_str not in episode_action_history_dict:
                        episode_action_history_dict[action_history_str] = 0
                    episode_action_history_dict[action_history_str] += 1
                print('episode_action_history_dict:', episode_action_history_dict)
            elif _step == self.config.env.max_steps - 1:
                for i in range(batch_size):
                    action_history = infos[i]['action_history']
                    action_history_str = ','.join(action_history)
                    if action_history_str not in episode_action_history_dict:
                        episode_action_history_dict[action_history_str] = 0
                    episode_action_history_dict[action_history_str] += 1
                print('episode_action_history_dict:', episode_action_history_dict)
            #batch.non_tensor_batch['episode_action_count_dict_sum'] = episode_action_count_dict_sum
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        for i in range(batch_size):
            _save_one_trajectory(
                id = i,
                traj_uid=str(traj_uid[i]),
                question=question_list[i]['question'],
                answer=question_list[i]['answer'],
                history_steps=[
                    {
                        "step": s,
                        "action": self.tokenizer.decode(
                            total_batch_list[i][s]['responses'], skip_special_tokens=True
                        ),
                        "reward": float(total_batch_list[i][s]['rewards']),
                        "active": bool(total_batch_list[i][s]['active_masks']),
                    }
                    for s in range(len(total_batch_list[i]))
                ],
                infos=[str(info) for info in total_infos[i]],
                blackboard = obs['anchor'][i]
            )

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, episode_input_token_num, episode_output_token_num, episode_action_history_dict
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, total_input_token_num, total_output_token_num, total_action_history_dict = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid = filter_group_data(batch_list=batch_list,
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_input_token_num, total_output_token_num, total_action_history_dict = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            episode_input_token_num=total_input_token_num,
            episode_output_token_num=total_output_token_num,
            episode_action_history_dict=total_action_history_dict,
        )
        
        return gen_batch_output
