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

import json
import re
import string
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import gymnasium as gym
import numpy as np
import tiktoken
from word2number import w2n

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from agent import Agent, ToolAgent, ReasonAgent, SpecializedToolAgent
from prompt_lib import init_role_prompt, role_blackboard_prompt, generate_expert, init_role_list, role_blackboard_empty_prompt

judge_prompt = '''
### System Role
You are an intelligent and impartial evaluator. Your task is to compare an AI model's prediction against the ground truth answer.

### Input Data
1. **Question:** {question}
2. **Ground Truth:** {answer}
3. **Model Prediction:** {output}

### Evaluation Criteria
Please assess whether the "Model Prediction" conveys the same meaning as the "Ground Truth".
- Focus on the semantic content. Do not penalize for differences in phrasing, punctuation, or capitalization.
- If the prediction contains the core information present in the ground truth and does not add contradictory information, mark it as **CORRECT**.
- If the prediction is missing key facts or contains wrong information, mark it as **WRONG**.

### Output Format
1. First, provide a brief reasoning explaining your judgment (1-2 sentences).
2. Then, output the final label: "CORRECT" or "WRONG".

### Your Evaluation
**Reasoning:**
[Your reasoning here]

**Label:**
[CORRECT / WRONG]
'''

class BlackboardEnv(gym.Env):
    """
    一个模拟多智能体通过共享黑板协作解决问题的环境。
    我们训练的“指挥家”LLM 的动作是选择下一个要发言的专家智能体。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化环境。
        
        :param config: 一个包含环境配置的字典，通常来自 YAML 文件。
                       - max_steps: 一个回合的最大步数。
                       - agents: 一个描述专家智能体名称的列表。
                       - tasks: 一个包含任务信息的列表，每个任务是一个字典，包含任务描述和答案
        """
        self.config = config
        self.seed = config.get("seed", 42)
        #np.random.seed(self.seed)
        self.max_steps = config.get("max_steps", 10)
        self.agent_names = config.get("available_agents", [])
        self.agent_group = []
        for agent_name in self.agent_names:
            if agent_name == "web_search" or agent_name == "python_calculate" or agent_name == "knowledge_retrieve":
                self.agent_group.append(SpecializedToolAgent(role=agent_name, mtype="deepseek-chat", temperature=0.1))
            elif agent_name == "reason":
                self.agent_group.append(ReasonAgent(role=agent_name, mtype="deepseek-chat", temperature=0.1))
            else:
                self.agent_group.append(Agent(role=agent_name, mtype="deepseek-chat", temperature=0.1))
        self.num_agents = len(self.agent_names)
        self.tasks = config.get("tasks", [])
        #初始化随机选择一个task
        #self.current_task_idx = 0
        self.current_task_idx = np.random.randint(0, len(self.tasks)-1)
        self.task_data = self.tasks[self.current_task_idx]
        self.ground_truth = self.task_data.get("answer", "")
        for agent in self.agent_group:
            #print(agent.role)
            if agent.role == "expert" or agent.role == "expert2":
                expert1, expert2 = generate_expert(self.task_data['task'])
                self.expert_role = expert1[0]
                self.expert2_role = expert2[0]
                init_prompt = init_role_prompt["expert"].format(role=self.expert_role, roles_description=expert1[1], question=self.task_data['task'])
            elif agent.role == 'web_search' or agent.role == 'knowledge_retrieve' or agent.role == 'python_calculate':
                init_prompt = init_role_prompt["web_search"]
            else:
                init_prompt = init_role_prompt[agent.role].format(question=self.task_data['task'])
            agent.set_meta_prompt(init_prompt)
        # 环境状态变量
        self.blackboard: Dict[str, Any] = {"knowledge": [], "discussions about the problem": [], "tasks": [self.task_data['task']]}  # 黑板内容
        #self.blackboard: Dict[str, Any] = {"discussions about the problem": [], "tasks": [self.task_data['task']]}  # 黑板内容
        self.current_step: int = 0
        self.ground_truth: str = ""

        self.correct = 0
        self.input_token_num = 0
        self.output_token_num = 0

    def normalize_answer(self, s):

        def normalize_numbers(text):
            # 尝试将文字数字转换为阿拉伯数字
            try:
                return str(w2n.word_to_num(text))
            except ValueError:
                return text

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        s = white_space_fix(remove_articles(remove_punc(lower(s))))
        tokens = s.split()
        tokens = [normalize_numbers(tok) for tok in tokens]

        return ' '.join(tokens)


    def f1_score(self, prediction, ground_truth):
        normalized_prediction = self.normalize_answer(prediction)
        normalized_ground_truth = self.normalize_answer(ground_truth)

        ZERO_METRIC = (0, 0, 0)

        if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            return ZERO_METRIC
        if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            return ZERO_METRIC

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return ZERO_METRIC
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall

    def process_aswer(self, answer: str) -> str:
        """
        处理专家智能体的回答，提取最终答案。
        """
        # 寻找 "the final answer is" 后面的内容
        pattern = r'the final answer is\s*[:：]?\s*(?:boxed\[(.*?)\]|(\S+))'
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            extracted_answer = match.group(1) if match.group(1) is not None else match.group(2)
            return extracted_answer.strip()
        else:
            # 如果没有找到，返回整个回答
            return answer.strip()


    def _get_observation(self) -> Dict[str, Any]:
        """构建并返回当前的观测。"""
        # valid_actions_str = ", ".join([f"{i}: {name}" for i, name in enumerate(self.agent_names)])
        return self.blackboard
    
    def get_info(self) -> Dict[str, Any]:
        """获取当前环境的额外信息。"""
        return {
            "current_step": self.current_step,
            "current_task": self.task_data['task'],
            "available_actions": init_role_list,
            "ground_truth": self.ground_truth,
            "correct": self.correct,
            "input_token_num": self.input_token_num,
            "output_token_num": self.output_token_num,
            "previous_action": self.previous_action,
            "previous_output": self.previous_output,
            "action_count_dict": self.action_count_dict,
            "action_history": self.action_history
        }

    def _execute_agent_action(self, action: str):
        """模拟执行一个专家智能体的动作，并返回其贡献。"""
        #print('Executing action:', action)
        agent_index = self.agent_names.index(action.lower())
        agent_function = self.agent_names[agent_index]
        for temp_agent in self.agent_group:
            if agent_function == temp_agent.role.lower():
                if self.current_step == 0:
                    if agent_function == 'terminate':
                        role_prompt = role_blackboard_empty_prompt[temp_agent.role].format(blackboard = self.blackboard, decision_prompt=self.decision_prompt,question=self.task_data['task'])
                    elif agent_function == 'direct_answer':
                        role_prompt = role_blackboard_empty_prompt[temp_agent.role].format(blackboard = self.blackboard,question=self.task_data['task'], decision_prompt=self.decision_prompt)
                    # elif agent_function == 'debate':
                    #     debate_list = ['reasoning','expert','critique','arbitrate']
                    #     for debate_agent in self.agent_group:
                    #         if debate_agent.role.lower() in debate_list and debate_agent.role.lower()+':' in self.blackboard['discussions about the problem']:
                    #             temp_role_prompt = role_blackboard_empty_prompt[debate_agent.role.lower()].format(blackboard = self.blackboard,question=self.task_data['task'])
                    #             temp_role_prompt = temp_role_prompt+'It\'s not necessary to fully agree with each other\'s perspectives, as our objective is to find the correct answer.'
                    #             temp_input_context = temp_agent.preprocess(self.blackboard, temp_role_prompt)
                    #             debate_content = debate_agent.get_answer(temp_input_context[0])
                    #             start_tag = "<output>"
                    #             end_tag = "</output>"
                    #             start_idx = content.find(start_tag)
                    #             end_idx = content.find(end_tag)
                    #             if start_idx == -1 or end_idx == -1:
                    #                 # If we can't find a valid <output>...</output> block, return the whole content
                    #                 extracted_output = content.strip()
                    #             else:
                    #                 extracted_output = content[start_idx + len(start_tag):end_idx].strip().lower()
                    #             self.blackboard['discussions about the problem'] = [debate_agent.role.lower()+':'+extracted_output]
                    else:
                        role_prompt = role_blackboard_empty_prompt[temp_agent.role].format(blackboard = self.blackboard,question=self.task_data['task'])
                else:
                    if agent_function == 'terminate':
                        role_prompt = role_blackboard_prompt[temp_agent.role].format(blackboard = self.blackboard, decision_prompt=self.decision_prompt,question=self.task_data['task'])
                    elif agent_function == 'direct_answer':
                        role_prompt = role_blackboard_prompt[temp_agent.role].format(blackboard = self.blackboard,question=self.task_data['task'], decision_prompt=self.decision_prompt)
                    # elif agent_function == 'debate':
                    #     debate_list = ['reasoning','expert','critique','arbitrate']
                    #     for debate_agent in self.agent_group:
                    #         if debate_agent.role.lower() in debate_list and debate_agent.role.lower()+':' in self.blackboard['discussions about the problem']:
                    #             temp_role_prompt = role_blackboard_prompt[debate_agent.role.lower()].format(blackboard = self.blackboard,question=self.task_data['task'])
                    #             temp_role_prompt = temp_role_prompt+'It\'s not necessary to fully agree with each other\'s perspectives, as our objective is to find the correct answer.'
                    #             temp_input_context = temp_agent.preprocess(self.blackboard, temp_role_prompt)
                    #             debate_content = debate_agent.get_answer(temp_input_context[0])
                    #             start_tag = "<output>"
                    #             end_tag = "</output>"
                    #             start_idx = content.find(start_tag)
                    #             end_idx = content.find(end_tag)
                    #             if start_idx == -1 or end_idx == -1:
                    #                 # If we can't find a valid <output>...</output> block, return the whole content
                    #                 extracted_output = content.strip()
                    #             else:
                    #                 extracted_output = content[start_idx + len(start_tag):end_idx].strip().lower()
                    #             self.blackboard['discussions about the problem'] = [debate_agent.role.lower()+':'+extracted_output]
                    else:
                        role_prompt = role_blackboard_prompt[temp_agent.role].format(blackboard = self.blackboard,question=self.task_data['task'])
                input_context = temp_agent.preprocess(self.blackboard, role_prompt)

                # encoding = tiktoken.get_encoding("cl100k_base")#使用同一编码方式计算token
                # input_ids = encoding.encode(input_context[0]["content"])
                # self.input_token_num += len(input_ids)

                content, prompt_tokens, completion_tokens = temp_agent.get_answer(input_context[0])
                self.input_token_num += prompt_tokens
                self.output_token_num += completion_tokens
                input_length_ratio = prompt_tokens / 32768
                output_length_ratio = completion_tokens / 8192  # 假设4096是模型的最大上下文长度#########################

                # output_ids = encoding.encode(content)
                # self.output_token_num += len(output_ids)

                #print('Agent content:', content)
                #print((str(temp_agent.role)+ '**'+content).encode('utf-8'))
                if "waiting for more information" in str(content).lower():
                    return "No need for my contribution.",input_length_ratio, output_length_ratio

###############
                encoding = tiktoken.get_encoding("cl100k_base")
                think_start_tag = "<thought>"
                think_end_tag = "</thought>"
                think_start_idx = content.find(think_start_tag)
                think_end_idx = content.find(think_end_tag)
                start_tag = "<output>"
                end_tag = "</output>"
                start_idx = content.find(start_tag)
                end_idx = content.find(end_tag)
                if start_idx == -1 or end_idx == -1:
                    # If we can't find a valid <output>...</output> block, return the whole content
                    extracted_output = content.strip()
                    #对长度大于4096的进行截断
                    if len(encoding.encode(extracted_output)) > 4096 and len(encoding.encode(extracted_output)) < 8192:
                        print("Output tokens exceed 4096, truncating...")
                        encoded_output = encoding.encode(extracted_output)
                        truncated_output = encoding.decode(encoded_output[-4096:])
                        extracted_output = truncated_output
                    else:
                        encoded_output = encoding.encode(extracted_output)
                        truncated_output = encoding.decode(encoded_output[:4096])
                        extracted_output = truncated_output
                else:
                    if think_start_idx != -1 and think_end_idx != -1 and think_start_idx < start_idx:
                        #print('Found thought tags in the content.', agent_function)
                        extracted_thought = content[think_start_idx + len(think_start_tag):think_end_idx].strip()
                        #如果thought过长就不用thought
                        if len(encoding.encode(extracted_thought)) > 4096:
                            #print("Thought tokens exceed 4096, ignoring thought...")
                            extracted_output = content[start_idx + len(start_tag):end_idx].strip()
                        else:
                            extracted_output = "Thought: " + extracted_thought + "\nAnswer: " + content[start_idx + len(start_tag):end_idx].strip()
                    else:
                        extracted_output = content[start_idx + len(start_tag):end_idx].strip()
############

        if agent_function == "task_decompose":
            self.blackboard['discussions about the problem'].append(str(agent_function)+':'+extracted_output)
        elif agent_function == "expert":
            if self.current_step == 0:
                expert_prompt = role_blackboard_empty_prompt['expert'].format(blackboard = self.blackboard,question=self.task_data['task'])
            else:
                expert_prompt = role_blackboard_prompt['expert'].format(blackboard = self.blackboard,question=self.task_data['task'])
            self.blackboard['discussions about the problem'].append(str(self.expert_role)+':'+extracted_output)
            for tmp_agent in self.agent_group:
                if tmp_agent.role == 'expert2':
                    input_context = tmp_agent.preprocess(self.blackboard, expert_prompt)
                    content, prompt_tokens, completion_tokens = tmp_agent.get_answer(input_context[0])
                    think_start_tag = "<thought>"
                    think_end_tag = "</thought>"
                    think_start_idx = content.find(think_start_tag)
                    think_end_idx = content.rfind(think_end_tag)
                    start_tag = "<output>"
                    end_tag = "</output>"
                    start_idx = content.find(start_tag)
                    end_idx = content.find(end_tag)
                    if start_idx == -1 or end_idx == -1:
                        # If we can't find a valid <output>...</output> block, return the whole content
                        extracted_output2 = content.strip()
                        #对长度大于4096的进行截断
                        if len(encoding.encode(extracted_output2)) > 4096 and len(encoding.encode(extracted_output2)) < 8192:
                            #print("Output tokens exceed 4096, truncating...")
                            encoded_output = encoding.encode(extracted_output2)
                            truncated_output = encoding.decode(encoded_output[-4096:])
                            extracted_output2 = truncated_output
                        else:
                            encoded_output = encoding.encode(extracted_output2)
                            truncated_output = encoding.decode(encoded_output[:4096])
                            extracted_output2 = truncated_output
                    else:
                        if think_start_idx != -1 and think_end_idx != -1 and think_start_idx < start_idx:
                            #print('Found thought tags in the content.', agent_function)
                            extracted_thought2 = content[think_start_idx + len(think_start_tag):think_end_idx].strip()
                            #如果thought过长就不用thought
                            if len(encoding.encode(extracted_thought2)) > 4096:
                                #print("Thought tokens exceed 4096, ignoring thought...")
                                extracted_output2 = content[start_idx + len(start_tag):end_idx].strip()
                            else:
                                extracted_output2 = "Thought: " + extracted_thought2 + "\nAnswer: " + content[start_idx + len(start_tag):end_idx].strip()
                        else:
                            extracted_output2 = content[start_idx + len(start_tag):end_idx].strip()
                    self.blackboard['discussions about the problem'].append(str(self.expert2_role)+':'+extracted_output2)
            extracted_output = 'expert1:' + extracted_output + '\nexpert2:' + extracted_output2
        elif agent_function == "reason":
            self.blackboard['discussions about the problem'].append(str(agent_function)+':'+extracted_output)
        elif agent_function == "direct_answer":
            self.blackboard['discussions about the problem'].append(str(agent_function)+':'+extracted_output)
            if "the final answer is" in content.lower():#####注意大小写匹配
                self.blackboard['discussions about the problem'].append('**solved**')
        elif agent_function == "arbitrate":
            self.blackboard['discussions about the problem'].append(str(agent_function)+':'+extracted_output)
        elif agent_function == "verify":
            self.blackboard['discussions about the problem'].append(str(agent_function)+':'+extracted_output)
        # elif agent_function == "clean":
        #     return "\n\n[清理者贡献]:\n# 删除了多余的注释和空行。\ndef hello_world():\n    print('Hello World')"
        elif agent_function == "web_search":
            self.blackboard['knowledge'].append(str(agent_function)+':'+extracted_output)
        elif agent_function == "knowledge_retrieve":
            self.blackboard['knowledge'].append(str(agent_function)+':'+extracted_output)
        elif agent_function == "python_calculate":
            self.blackboard['knowledge'].append(str(agent_function)+':'+extracted_output)
        elif agent_function == "terminate":
            self.blackboard['discussions about the problem'].append(str(agent_function)+':'+extracted_output)
            if "the final answer is" in content.lower():#####注意大小写匹配
                self.blackboard['discussions about the problem'].append('**solved**')
        elif agent_function == "critique":
            self.blackboard['discussions about the problem'].append(str(agent_function)+':'+extracted_output)
        elif agent_function == "summarize":
            self.blackboard['discussions about the problem'] = ["summary of previous discussions"+':'+extracted_output]
        elif agent_function == "question":
            self.blackboard['discussions about the problem'].append(str(agent_function)+':'+extracted_output)
        elif agent_function == "clean":
            modify_prompt = role_blackboard_prompt['modify'].format(blackboard = self.blackboard, suggestion=extracted_output)
            for tmp_agent in self.agent_group:
                if tmp_agent.role == 'modify':
                    input_context = tmp_agent.preprocess(self.blackboard, modify_prompt)
                    content, prompt_tokens, completion_tokens = tmp_agent.get_answer(input_context[0])
                    start_tag = "<output>"
                    end_tag = "</output>"
                    start_idx = content.find(start_tag)
                    end_idx = content.find(end_tag)
                    if start_idx == -1 or end_idx == -1:
                        # If we can't find a valid <output>...</output> block, return the whole content
                        extracted_output = content.strip()
                    else:
                        extracted_output = content[start_idx + len(start_tag):end_idx].strip().lower()
                    self.blackboard['discussions about the problem'] = [extracted_output]
        elif agent_function == 'debate':
            self.blackboard['discussions about the problem'].append(str(agent_function)+':'+extracted_output)
        else:
            self.blackboard['discussions about the problem'].append(str(agent_function)+':'+extracted_output)
        
        encoding = tiktoken.get_encoding("cl100k_base")
        blackboard_str = json.dumps(self.blackboard, ensure_ascii=False)
        token_count = len(encoding.encode(blackboard_str))
        max_token_len = 20000

        if token_count > max_token_len:
            print(f"[INFO] Blackboard length {len(blackboard_str)} exceeds max_len {max_token_len}, calling summary agent...")
            for tmp_agent in self.agent_group:
                if tmp_agent.role == 'summarize':
                    # 构造 summary prompt
                    summary_prompt = role_blackboard_prompt['summarize'].format(
                        blackboard=self.blackboard,
                        question=self.task_data['task']
                    )
                    input_context = tmp_agent.preprocess(self.blackboard, summary_prompt)
                    summary_content, prompt_tokens, completion_tokens = tmp_agent.get_answer(input_context[0])
                    #self.input_token_num += prompt_tokens
                    #self.output_token_num += completion_tokens

                    start_idx = summary_content.find(start_tag)
                    end_idx = summary_content.find(end_tag)
                    if start_idx == -1 or end_idx == -1:
                        summary_output = summary_content.strip()
                    else:
                        summary_output = summary_content[start_idx + len(start_tag):end_idx].strip()
                    #对超过30000的进行强制截断
                    if len(encoding.encode(summary_output)) > 20000:
                        summary_output = encoding.decode(encoding.encode(summary_output)[:20000])
                    # 用摘要替换原黑板的 discussions
                    self.blackboard['discussions about the problem'] = [f"summary of previous discussions:{summary_output}"]
                    print("[INFO] Blackboard content summarized.")
                    break
            if agent_function == 'terminate':
                if "the final answer is" in content.lower():#####注意大小写匹配
                    self.blackboard['discussions about the problem'].append(str(agent_function)+':'+extracted_output)
                    self.blackboard['discussions about the problem'].append('**solved**')
        return extracted_output,input_length_ratio, output_length_ratio

    def reset(self, seed: Optional[int] = None, question: Dict = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        重置环境到一个新的回合。
        
        :param task_info: (可选) 包含任务信息的字典，框架可能会传入。
        :return: 初始观测。
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        

        self.current_step = 0
        self.done = False
        self.correct = 0
        self.input_token_num = 0
        self.output_token_num = 0
        self.action_history = []
        self.previous_action = "None"
        self.previous_output = "None"
        self.action_count_dict = {}
        self.process_reward = 0.0
        for agent_name in self.agent_names:
            self.action_count_dict[agent_name] = 0
        
        # 随机选取任务列表中的任务
        if question is not None:
            #print('Using provided question for reset.')
            self.task_data = {"task": question['question'], "answer": question['answer']}
            self.decision_prompt = question['decision_prompt']
            self.answer_type = question['answer_type']
        # else:
        #     self.current_task_idx += 1
        #     self.current_task_idx = np.random.randint(0, len(self.tasks)-1)
        #     self.task_data = self.tasks[self.current_task_idx-1]
        #     self.decision_prompt = 'output the final answer with your answer in the form **{{the final answer is boxed[answer]}}**, at the end of your response.'
        
        self.blackboard = {"knowledge": [], "discussions about the problem": [], "tasks": [self.task_data['task']]}
        #self.blackboard = {"discussions about the problem": [], "tasks": [self.task_data['task']]}
        self.ground_truth = self.task_data.get("answer", "")
        expert1, expert2 = generate_expert(self.task_data['task'])
        self.expert_role = expert1[0]
        #print('Generated expert role:', self.expert_role)
        self.expert2_role = expert2[0]
        #print('Generated expert2 role:', self.expert2_role)
        for agent in self.agent_group:
            if agent.role == "expert":
                #self.expert_role = expert1[0]
                init_prompt = init_role_prompt["expert"].format(role=self.expert_role, roles_description=expert1[1], question=self.task_data['task'])
            elif agent.role == "expert2":
                #self.expert2_role = expert2[0]
                init_prompt = init_role_prompt["expert"].format(role=self.expert2_role, roles_description=expert2[1], question=self.task_data['task'])
            elif agent.role == 'web_search' or agent.role == 'knowledge_retrieve' or agent.role == 'python_calculate':
                init_prompt = init_role_prompt["web_search"]
            else:
                init_prompt = init_role_prompt[agent.role].format(question=self.task_data['task'])
            agent.set_meta_prompt(init_prompt)
            
        
        return self._get_observation(), self.get_info()

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        在环境中执行一个时间步。
        
        :param action: 指挥家 LLM 选择的专家智能体名称。
        :return: (observation, reward, done, info) 元组。
        """
        if self.done:
            print("Step called on a terminated episode. Please reset the environment.")
            return self._get_observation(), 0.0, self.done, self.get_info()
        #print('Chosen action:', action)
        if action not in self.agent_names:
            action_str = f"\n\n[错误：选择了无效的动作 {action}]"
            print(f"\n\n[错误：选择了无效的动作 {action}]")
            self.previous_action = action
            self.previous_output = 'Error: Invalid action selected'
            self.current_step += 1
            reward, self.done = -0.01, False#######################################
            #self.blackboard += action_str
            return self._get_observation(), reward, self.done, self.get_info()

        # 1. 执行动作并更新黑板状态
        contribution,input_length_ratio, output_length_ratio = self._execute_agent_action(action)
        # print('blackboard:', self.blackboard)
        # print('Agent contribution:', contribution)
        self.previous_action = action
        self.previous_output = contribution
        
        self.current_step += 1
        self.action_history.append(action)
        self.action_count_dict[action] += 1

        # 2. 计算过程奖励
        # 2.1 长度惩罚：鼓励用更少的步骤完成任务
        length_penalty = -0.05 * (input_length_ratio+output_length_ratio)
        #print('Length penalty:', length_penalty)
        #length_penalty = 0.0
        
        # 2.2 正确选择奖励/惩罚：如果智能体拒绝执行，则给予负奖励
        if contribution.strip() == "No need for my contribution.":
            is_correct_choice = False
            #correct_choice_reward = -0.02
            correct_choice_reward = 0.0
        else:
            is_correct_choice = True
            correct_choice_reward = 0
        
            #correct_choice_reward += 2.0

        #process_reward = length_penalty + correct_choice_reward
        self.process_reward += length_penalty + correct_choice_reward
        output_reward = 0.0
        output_reward += length_penalty + correct_choice_reward

        # 3. 判断回合是否结束并计算结果奖励
        #self.done = False
        #agent_name =  self.agent_names[self.agent_names.index(action.lower())]
        agent_name = action.lower()
        if "**solved**" in self.blackboard['discussions about the problem']:
            self.done = True
            #计算结果奖励
            if self.answer_type == 'multiple':
                if "boxed" in contribution:
                    #print('Final contribution:', contribution)
                    start_idx = contribution.rfind("boxed") + len("boxed")
                    #end_idx = contribution.find("}", start_idx)
                    final_answer = contribution[start_idx:].strip()
                    option = re.findall(r'[A-Z]',final_answer)
                    if len(option) == 0:
                        option = re.findall(r'[a-z]',final_answer)
                    if len(option) != 0 and option[0].lower() == self.ground_truth[0].lower():
                        result_reward = 1.0
                        self.correct = 1
                    # elif self.ground_truth[2:].strip().lower() in final_answer.lower():
                    #     result_reward = 1.0
                    #     self.correct = 1
                    else:
                        result_reward = -0.1
                else:
                    result_reward = -0.1########################################
                    # if len(self.ground_truth[2:].strip()) > 1 and self.ground_truth[2:].strip().lower() in contribution.lower():
                    #     result_reward = 1.0
                    #     self.correct = 1
                    # else:
                    #     result_reward = 0.0
            elif self.answer_type == 'single':
                #匹配contribution中”the final answer is"后面的内容作为答案
                if "boxed" in contribution.lower():
                    start_idx = contribution.lower().rfind("boxed") + len("boxed")
                    final_answer = contribution[start_idx:].strip()
                    #去掉括号
                    if final_answer.startswith("{") and final_answer.endswith("}"):
                        final_answer = final_answer[1:-1].strip()
                    # print('blackboard:', self.blackboard)
                    # print('action history:', self.action_history)
                    # print('final answer:', final_answer)
                    # print('ground truth:', self.ground_truth)
                    f1, precision, recall = self.f1_score(final_answer, self.ground_truth)
                    # if self.ground_truth.strip().lower() in final_answer.lower():
                    #     result_reward = 1.0
                    #     self.correct = 1
                    # else:
                    #     result_reward = 0.0
                    # print('Final answer:', final_answer)
                    # print('Ground truth:', self.ground_truth)
                    # print(f'F1: {f1}, Precision: {precision}, Recall: {recall}')
                    result_reward = f1
                    if f1 == 1.0:
                        self.correct = 1
                    else:
                        judge_input = {"role": "user", "content": judge_prompt.format(question=self.task_data['task'],answer=self.ground_truth,output=final_answer)}
                        from utils import generate_answer
                        judge_output = generate_answer([judge_input], 'gemini-3-flash-preview', None, temperature=0.0, top_p=0.9)
                        judge_output_text = judge_output.choices[0].message.content
                        if '**Label:**' in judge_output_text:
                            label = judge_output_text.split('**Label:**')[-1].strip().lower()
                            if 'correct' in label.lower():
                                self.correct = 1
                                result_reward = 1.0
                            else:
                                print('预测错误，正确答案:',self.ground_truth,'预测答案:',final_answer)
                                result_reward = 0.0
                        else:
                            print('预测错误，正确答案:',self.ground_truth,'预测答案:',final_answer)
                            result_reward = 0.0

                else:
                    print('No boxed answer found in contribution:', contribution)
                    result_reward = 0.0

                    # result_reward = 0.0########################################

            self.action_history.append(str(self.correct))##################
            output_reward += result_reward
            # if result_reward > 0:
            #     output_reward += self.process_reward  # 如果回答正确，给予过程奖励作为额外奖励
        elif self.current_step >= self.max_steps:
            #回合结束但未解决问题，给予较大负奖励
            print('达到最大轮数')
            result_reward = 0.0
            
            from utils import generate_answer
            answer_prompt = {"role": "user", "content": 'The blackboard contains discussions of a group of action agents:'+ str(self.blackboard)+'\nCritically evaluate content on the blackboard: check for correctness, clarity, and consistency, and answer the question as accurately as possible: ' + self.task_data['task'] + '\n'+self.decision_prompt}
            final_answer = generate_answer([answer_prompt], 'deepseek-chat', None, temperature=0.1, top_p=0.9)
            contribution = final_answer.choices[0].message.content
            self.blackboard['discussions about the problem'].append('\nmax_loop,answer:'+contribution)
            #计算结果奖励
            if self.answer_type == 'multiple':
                if "boxed" in contribution:
                    #print('Final contribution:', contribution)
                    start_idx = contribution.rfind("boxed") + len("boxed")
                    #end_idx = contribution.find("}", start_idx)
                    final_answer = contribution[start_idx:].strip()
                    option = re.findall(r'[A-Z]',final_answer)
                    if len(option) == 0:
                        option = re.findall(r'[a-z]',final_answer)
                    if len(option) != 0 and option[0].lower() == self.ground_truth[0].lower():
                        self.correct = 1
                        result_reward = 1.0
                    # elif self.ground_truth[2:].strip().lower() in final_answer.lower():
                    #     result_reward = 1.0
                    #     self.correct = 1
                    else:
                        result_reward = -0.1
                else:
                    if self.ground_truth[2:].strip().lower() == contribution.lower():
                        print('没有boxed但答案正确',contribution, "  vs  ",self.ground_truth)
                        self.correct = 1
                        result_reward = 1.0
                    else:
                        result_reward = -0.1
            elif self.answer_type == 'single':
                #匹配contribution中”the final answer is"后面的内容作为答案
                if "boxed" in contribution.lower():
                    start_idx = contribution.lower().rfind("boxed") + len("boxed")
                    final_answer = contribution[start_idx:].strip()
                    #去掉括号
                    if final_answer.startswith('{') and final_answer.endswith('}'):
                        final_answer = final_answer[1:-1].strip()
                    f1, precision, recall = self.f1_score(final_answer, self.ground_truth)
                    # if self.ground_truth.strip().lower() in final_answer.lower():
                    #     result_reward = 1.0
                    #     self.correct = 1
                    # else:
                    #     result_reward = 0.0
                    # print('Final answer:', final_answer)
                    # print('Ground truth:', self.ground_truth)
                    # print(f'F1: {f1}, Precision: {precision}, Recall: {recall}')
                    #result_reward = f1
                    if f1 == 1.0:
                        self.correct = 1
                        result_reward = 1.0
                    else:
                        judge_input = {"role": "user", "content": judge_prompt.format(question=self.task_data['task'],answer=self.ground_truth,output=final_answer)}
                        judge_output = generate_answer([judge_input], 'gemini-3-flash-preview', None, temperature=0.0, top_p=0.9)
                        judge_output_text = judge_output.choices[0].message.content
                        print('judge_output_text',judge_output_text)
                        if '**Label:**' in judge_output_text:
                            label = judge_output_text.split('**Label:**')[-1].strip().lower()
                            if 'correct' in label.lower():
                                self.correct = 1
                                result_reward = 1.0
                            else:
                                result_reward = 0.0
                        else:
                            result_reward = 0.0
                else:
                    print('No boxed in contribution:', contribution)
                    result_reward = 0.0
                    # f1, precision, recall = self.f1_score(contribution, self.ground_truth)
                    # #result_reward = f1
                    # if f1 == 1.0:
                    #     self.correct = 1
                    #     result_reward = 1.0
                #result_reward = 0.0###################先不给奖励
            self.action_history.append(str(self.correct))##################
            output_reward += result_reward
            # if result_reward > 0:
            #     output_reward += self.process_reward  # 如果回答正确，给予过程奖励作为额外奖励


        # 4. 构建 info 字典，用于传递额外信息
        info = {
            "step": self.current_step,
            "current_task": self.task_data['task'],
            "available_actions": init_role_list,
            "ground_truth": self.ground_truth,
            "correct": self.correct,
            "input_token_num": self.input_token_num,
            "output_token_num": self.output_token_num,
            "previous_action": self.previous_action,
            "previous_output": self.previous_output,
            "action_count_dict": self.action_count_dict,
            "action_history": self.action_history
        }

        return self._get_observation(), output_reward, self.done, info
