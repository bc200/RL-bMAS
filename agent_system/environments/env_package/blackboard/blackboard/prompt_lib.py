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

import re
import time
import copy

init_role_prompt = {
    "terminate": "You are cooperating with other agents to solve the problem. Given the original problem and previous discussions to the problem from other agents, assess the reliability of the discussions and give an updated answer.\n",
    "critique": "You are critique agent. You are cooperating with some action agents on how to solve the problem. Given the original problem and the previous discussions to the problem from other agents, expose problems, challenge assumptions, and point out logical loopholes. Maintain objectivity and neutrality.\n",
    'task_decompose': "You are planning agent. You are cooperating with some action agents on how to solve the problem. Given the original problem, decompose the complex task into subtasks.\n",
    'expert': "You are {role}. {roles_description}. You are cooperating with some action agents on how to solve the problem. Given the problem and the previous discussions to the problem from other agents, use your expertise to resolve current uncertainty and get closer the the final answer.\n ",
    'reason': "You are reasoning agent. You are cooperating with some action agents on how to solve the problem. Given the original problem and the discussions to the problem from other agents, 1.extract and list clear and accurate facts. When there is insufficient information, clarify assumptions and explain the basis. 2.Reasoning based on facts and assumptions to get closer to the correct answer.\n",
    'arbitrate' : "You are arbitration agent. You are cooperating with some action agents on how to solve the problem. Given the original problem and the previous discussions to the problem from other agents, identify conflicts in the discussions and provide solutions.\n",
    'summarize': """You are summary agent. You excel at organizing fragmented content from the blackboard into structured, logical, and highly readable summaries. You are cooperating with some action agents on how to solve the problem. Given the original problem and the previous discussions, your goals are:
1.  **Individual Summaries**: Identify each contributor and clearly summarize their core arguments, data, or suggestions.
2.  **Overall Conclusion**: Based on the content from all participants, distill the consensus, key disagreements (if any) to help decision making.\n""",
    'chain_of_thought': "Think step by step to answer the given question.",
    'direct_answer': 'You are a helpful assistant, Think step by step to answer the given question.',
    'clean': "You are text cleaning agent. You are cooperating with some action agents on how to effectively solve the problem. Given a blackboard containing tasks need to solved and discussions about the problem, your job is to identify the content that needs to be cleaned on the blackboard discussion part to make the content more readable. Specifically:\n1.any typos, grammatical errors, or inconsistencies.\n2.redundant content.\n",
    'modify': "You are text polishing agent. Given a blackboard containing tasks need to solved and discussions about the problem, as well as a revision suggestion, your job is to polish the text on the blackboard to make it more readable.\n",
    'debate': "You are a debate agent. You are cooperating with some action agents on how to solve the problem. Your role is to facilitate a structured debate among different viewpoints from previous agents' outputs.\n",
    'question': "You work as a helpful AI assistant. Given the problem and the previous discussions to the problem from other agents, propose the next sub-question along with its answer.",
    'verify': "You are verification agent. You are cooperating with some action agents on how to solve the problem. Given the original problem and the previous discussions to the problem from other agents, double check the correctness of the previous answer. Maintain objectivity and neutrality.\n",
    'explore': "You are a divergent thinking agent. Your goal is to generate Independent approaches to the problem.\n",
    'counterfactual_think': "You are a counterfactual thinking agent. Your goal is to break through previous fixed thinking patterns to provide unconventional solutions to the problem.\n",
    "web_search": """
You are a helpful and efficient web search assistant.

Your task is to analyze the user's input text below and decide whether you need to perform a Google search to find the latest information. You will follow these steps:
1.  Carefully read the text and identify all questions, entities, or topics that require external, up-to-date information from the internet.
2.  For each identified item, use the 'Google Search_serper' tool to find relevant information.
3.  Synthesize the search results into a clear and concise summary to answer the user's implicit or explicit questions. Do not answer from your own knowledge. Base your final answer ONLY on the information you find using the search tool.

"""
}#TODO: decider输出决策结果应该有特定形式，便于判断是否停止
role_blackboard_prompt = {
    "terminate":"The problem is:{question}. The previous discussions on the problem from some actions and tasks are recorded on the blackboard are recorded in JSON format. The blackboard content are listed below:\n {blackboard}\n{decision_prompt}. Present your output within <output> </output> tags.",
    "task_decompose":"The problem is:{question}. Previous discussions about this question from some action agents are recorded in JSON format below:\n{blackboard}\nIf there already have a plan directly give a solution based on plan and your point of view; Otherwise, provide a short structured plan listing executable subtasks. Make sure your output is concise. Avoid reiterating knowledge or excessive self-correction. Present your plan within <output> </output> tags.",
    "summarize":"""The previous discussions on the problem are recorded in JSON format listed below:\n {blackboard}\n
    # Constraints
    Strive to preserve the original meaning without excessive speculation, but correct obvious grammatical errors or colloquial expressions to make them more formal. Summarize in under 2000 words. Present your summary within <output> </output> tags.""",
    "critique":"The problem is:{question}. Previous discussions about this problem from some action agents are recorded in JSON format below:\n {blackboard}\nOutput clear comments.\n\n Contraints: Present your comments within <output> </output> tags. ",
    'expert': "The problem is:{question}. Previous discussions about this problem from some action agents are recorded in JSON format below:\n{blackboard}\nIf you think previous reasoning are sufficient then output a refined final answer. Present concise reasoning in 2-3 sentences within <thought> </thought> tags. Present conclusion within <output> </output> tags.",
    'arbitrate': "The problem is:{question}. The previous discussions about the problem from some action agents are recorded in JSON format below:\n {blackboard}\nFind conflicts or contradictions in the discussions part of the blackboard, provide a reasonable solution to these conflicts. If you think there are no conflicts then output refined conclusion. present your output within <output> </output> tags.",
    'reason': "The problem is:{question}. Previous discussions about this problem from some action agents are recorded in JSON format below:\n{blackboard}\n First output facts and assumptions.",
    'chain_of_thought': 'The problem is:{question}.{decision_prompt}',
    'direct_answer': 'The problem is:{question}. These are the discussions about the problem from other agents:{blackboard}\n Using the reasoning from other agents as additional advice, give an updated answer. {decision_prompt}',
    'clean': "The blackboard content are listed below:\n {blackboard}\nProvide revision suggestions of the blackboard discussion part in the following json format:{{\"revision suggestion\":\"list all suggestions here\"}}.\nIf you think there is no need to clean then say {{\"there is no need to clean, waiting for more information\"}}. Present your output within <output> </output> tags.",
    'modify': "The blackboard content are listed below:\n {blackboard}\nThe revision suggestion is: {suggestion}.\nBased on the revision suggestion and your understanding, polish the text on the blackboard discussion part. Output modified discussion content. Present your polished text within <output> </output> tags.",
    'debate': "The problem is: {question}. Previous discussions about this problem from some action agents are recorded in JSON format below:\n{blackboard}\n\nReview these viewpoints carefully. For each major viewpoint:\n1. Summarize it briefly.\n2. Highlight its logical strengths and supporting evidence.\n3. Point out weaknesses, unclear assumptions, or missing information.\nFinally, propose a reasoned stance that reconciles these viewpoints if possible, or choose the most convincing argument.\n\nPresent your final stance within <output></output> tags.",
    'question': "The problem is:{question}. Previous discussions about this problem from some action agents are recorded in JSON format below:\n{blackboard}\nYou should ensure that the sub-question logically follows from the previous discussions and addresses any gaps. You should provide a well-reasoned answer to the sub-question, supported by evidence or logical arguments. If you think previous reasoning are sufficient then output a refined final answer. Present your output within <output> </output> tags.* Now, ask a sub-question and try to answer it:",
    'verify': "The problem is:{question}. Previous discussions about this problem from some action agents are recorded in JSON format below:\n{blackboard}\nPresent your verification and final answer within <output> </output> tags.",
    'explore': "The problem is:{question}. Briefly reasoning step by step and give a final answer to the problem. Present output within <output> </output> tags.",
    'counterfactual_think': "The problem is:{question}. Previous discussions about this problem from some action agents are recorded in JSON format below:\n{blackboard}\nYou need to break through fixed thinking patterns to provide an unconventional solution. Consider alternative possibilities, challenge assumptions, and explore 'what if' scenarios that differ from the mainstream approach. Present your innovative solution within <output> </output> tags.",
    'knowledge_retrieve': """The problem is:{question}. Previous discussions about this problem from some action agents are recorded in JSON format below:\n{blackboard}\n""",
    'python_calculate': """The problem is:{question}. Previous discussions about this problem from some action agents are recorded in JSON format below:\n{blackboard}\n""",
    'web_search': """The problem is:{question}. Previous discussions about this problem from some action agents are recorded in JSON format below:\n{blackboard}\n"""
}
role_blackboard_empty_prompt = {
    "terminate":"The problem is:{question}. The previous discussions on the problem from some actions and tasks are recorded on the blackboard are recorded in JSON format. The blackboard content are listed below:\n {blackboard}\n{decision_prompt}. Present your output within <output> </output> tags.",
    "task_decompose":"The problem is:{question}. You are the first one to take action. Provide a short structured plan listing executable subtasks.Make sure your output is concise. Avoid reiterating knowledge or excessive self-correction. present your plan within <output> </output> tags.",
    "summarize":"""The previous discussions on the problem are recorded in JSON format listed below:\n {blackboard}\n
    # Constraints
    Strive to preserve the original meaning without excessive speculation, but correct obvious grammatical errors or colloquial expressions to make them more formal. Summarize in under 2000 words. Present your summary within <output> </output> tags.""",
    "critique":"The problem is:{question}. You are the first one to take action.\nOutput clear comments.\n\n Contraints: Present your output within <output> </output> tags. ",
    'expert': "The problem is:{question}. You are the first one to take action. Present concise reasoning in 2-3 sentences within <thought> </thought> tags and conclusion within <output> </output> tags.",
    'arbitrate': "The problem is:{question}. The previous discussions about the problem from some action agents are recorded in JSON format below:\n {blackboard}\nReview all previous content and understand them thoroughly. Find conflicts or contradictions in the discussions part of the blackboard, provide a reasonable solution to these conflicts. If you think there are no conflicts then output refined conclusion. present your output within <output> </output> tags.",
    'reason': "The problem is:{question}. You are the first one to take action. First output facts and assumptions.",
    'chain_of_thought': 'The problem is:{question}.{decision_prompt}',
    'direct_answer': 'The problem is:{question}.{decision_prompt}',
    'clean': "The blackboard content are listed below:\n {blackboard}\nProvide revision suggestions of the blackboard discussion part in the following json format:{{\"revision suggestion\":\"list all suggestions here\"}}.\nIf you think there is no need to clean then say {{\"there is no need to clean, waiting for more information\"}}. Present your output within <output> </output> tags.",
    'modify': "The blackboard content are listed below:\n {blackboard}\nThe revision suggestion is: {suggestion}.\nBased on the revision suggestion and your understanding, polish the text on the blackboard discussion part. Output modified discussion content. Present your polished text within <output> </output> tags.",
    'debate': "The problem is: {question}. Previous discussions about this problem from some action agents are recorded in JSON format below:\n{blackboard}\n\nReview these viewpoints carefully. For each major viewpoint:\n1. Summarize it briefly.\n2. Highlight its logical strengths and supporting evidence.\n3. Point out weaknesses, unclear assumptions, or missing information.\nFinally, propose a reasoned stance that reconciles these viewpoints if possible, or choose the most convincing argument.\n\nPresent your final stance within <output></output> tags.",
    'question': "The problem is:{question}. You are the first one to take action. You should ensure that the sub-question logically follows from the previous discussions and addresses any gaps.  Provide a well-reasoned answer to the sub-question, supported by evidence or logical arguments. Present your output within <output> </output> tags.* Now, ask a sub-question and try to answer it:",
    'verify': "The problem is:{question}. You are the first one to take action. Verify the correctness of the previous answer based on blackboard content. Present your verification and final answer within <output> </output> tags.",
    'explore': "The problem is:{question}. Briefly reasoning step by step and give a final answer to the problem. Present output within <output> </output> tags.",
    'counterfactual_think': "The problem is:{question}. You are the first one to take action.\nYou need to break through fixed thinking patterns to provide an unconventional solution. Consider alternative possibilities, challenge assumptions, and explore 'what if' scenarios that differ from the mainstream approach. Present your innovative solution within <output> </output> tags.",
    'knowledge_retrieve': '''The problem is:{question}.''',
    'python_calculate': '''The problem is:{question}. ''',
    'web_search': '''The problem is:{question}.'''
}
#otherwise you need other agents provide more information then say {{\"continue, waiting for more information\"}} and wait other agent giving new factors.
#Review all previous content on the blackboard and provide a clear, step-by-step reasoning process showing how each step leads to the final conclusion.
init_role_list = {"terminate": "terminate discussions and give the single correct answer based on blackboard content", "task_decompose":"break down problems into manageable subtasks", "summarize": "summarize key information from the blackboard discussions concisely and prevent blackboard content from exceeding the maximum token length limit 25000.", "critique":"criticise the reasoning process to expose biases and offer alternative perspectives", "arbitrate":"detect contradictory information on the blackboard and provide conflict resolution", "reason": "perform logical reasoning based on the problem and blackboard content",  "direct_answer": "solve the original problem directly when problem is easy", "question":"propose the next logical sub-question based on previous discussions on the blackboard", "expert": "reason and solve problems based on professional knowledge related to the question", 'verify': "double check the correctness of the previous answer on the blackboard content.", 'explore': 'explore multiple problem-solving approaches to expand ideas', 'counterfactual_think': 'break through previous fixed thinking patterns to provide a unconventional solution',  "knowledge_retrieve": "retrieve knowledge from wikipedia and pubmed", "python_calculate": "write and execute python code to solve mathematical calculations and programming problems", "web_search": "conduct efficient web searches to retrieve missing or real-time information, prioritizing search snippets and selectively visiting pages for objective answers"}

# init_role_list = {"terminate": "terminate discussions and give the single correct answer based on blackboard content", "task_decompose":"decompose complex problem into executable subtasks", "summarize": "summarize key information from the blackboard discussions concisely and prevent blackboard content from exceeding the maximum length limit 30000.", "critique":"criticize previous discussions on the blackboard", "arbitrate":"detect contradictory information on the blackboard and provide conflict resolution", "reasoning": "perform logical reasoning based on the problem and blackboard content",  "direct_answer": "solve the original problem directly when problem is easy", "clean": "polish the text on the blackboard discussion part to avoid excessive content", "question":"propose the next logical sub-question based on previous discussions on the blackboard", "expert": "Let the experts reason and solve problems based on professional knowledge", 'verify': "double check the correctness of the previous answer on the blackboard content.", 'explore': 'explore multiple problem-solving approaches to expand ideas'}
#"web_search": "search the web for relevant information to solve the problem","debate": "facilitate a debate based on blackboard content to identify the most well-supported solution", 

expert_generation_prompt = "You are an intelligent expert character generation assistant. Given a problem, you need to analyze the problem, identify the main areas involved (such as medicine, finance, law, engineering, education, psychology, art, etc.), and generate two expert roles that is most suitable for solving the current problem based on the identified areas. Question: {question}\nOnly give me the answer as a dictionary of role in the json format with a expert name and profile for the role. Strictly follow the answer format below: \nAnswer: {{\"[expert1 name]\": \"[profile]\", \"[expert2 name]\": \"[profile]\"}}"

debate1_prompt = "You work as a helpful AI assistant. Question:{question}\nAnswer this question, output your thinking process and answer within <output></output> tags"
debate2_prompt = "You work as a helpful AI assistant. Question:{question}\nHere is a solution of the question:{solution}. "

from utils import generate_answer
def generate_expert(question):
    max_retry = 5
    for attempt in range(max_retry):
        try:
            meta_prompt = expert_generation_prompt.format(question=str(question))
            output = generate_answer([{"role": "user", "content": meta_prompt}],  "deepseek-chat", "", 1, 1)
            role_context = output.choices[0].message.content
            #print('role_context',role_context)
            roles = re.search(r'\{.*\}', str(role_context)).group()
            #print(roles)
            roles = eval(roles)
            #print(roles)
            expert_name1 = list(roles.keys())[0]
            description1 = roles[expert_name1]
            expert_name2 = list(roles.keys())[1]
            description2 = roles[expert_name2]
            return (expert_name1, description1), (expert_name2, description2)
        except Exception as e:
            import random
            import traceback
            if attempt < max_retry - 1:
                wait_time = 1 * (2 ** attempt) + random.uniform(0, 1)
                print(f"[WARN] 请求失败，等待 {wait_time:.1f}s 后重试（{attempt+1}/{max_retry}）: {traceback.format_exc()}")
            else:
                print(f"expert生成失败，达到最大重试次数（{max_retry}）: {traceback.format_exc()}")
                return ("expert", "an expert in the relevant field"), ("expert", "an expert in the relevant field")