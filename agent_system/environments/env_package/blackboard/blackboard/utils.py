import os

import openai
from openai import OpenAI, AsyncOpenAI
import time
import random
import pandas as pd
from glob import glob
import json
import jsonlines
import pyarrow.parquet as pq
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


def _get_api_key(explicit_api_key, environment_variable):
    """Resolve an API key without embedding credentials in source code."""
    api_key = explicit_api_key or os.getenv(environment_variable)
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set {environment_variable} in your local environment "
            "or pass api_key when constructing the agent."
        )
    return api_key



def generate_answer(answer_context, model, llm_ip=None, temperature=1, top_p=1, use_json=False):
    try:
        if model.find("gpt") < 0: # opensourced LLM
            # if model == 'Meta-Llama-3-8B-Instruct':
            #     #使用transformers库调用本地大模型
            #     tokenizer = AutoTokenizer.from_pretrained("../../llama3/LLM-Research/Meta-Llama-3-8B-Instruct")
            #     model = AutoModelForCausalLM.from_pretrained("../../llama3/LLM-Research/Meta-Llama-3-8B-Instruct")
            #     config = GenerationConfig(max_length=1024, num_beams=1, temperature=temperature, top_p=top_p)
            #     completion = model.generate(
            #         tokenizer.encode(answer_context, return_tensors="pt"),
            #         **config
            #     )
            #     completion = tokenizer.decode(completion[0], skip_special_tokens=True)

            if model == 'deepseek-r1-0528':
                #使用deerapi的api
                model = 'deepseek-r1-0528'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "DEER_API_KEY"),
                            base_url = os.getenv("DEER_BASE_URL", "https://api.deerapi.com/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    presence_penalty = 0.1,
                )
            if model == 'deepseek-chat':
                #使用deerapi的api
                model = 'deepseek-chat'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "DMX_API_KEY"),
                            base_url = os.getenv("DMX_BASE_URL", "https://www.dmxapi.com/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    presence_penalty = 0.1,
                    max_tokens=8192,
                )

            elif model == 'gemini-3-flash-preview':
                #使用deerapi的api
                model = 'gemini-3-flash-preview'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "DMX_API_KEY"),
                            base_url = os.getenv("DMX_BASE_URL", "https://www.dmxapi.com/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    max_tokens=4096,
                    presence_penalty = 0.1,
                )
                
            elif model == 'deepseek-v3.1':
                #使用deerapi的api
                model = 'deepseek-v3.1'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "DMX_API_KEY"),
                            base_url = os.getenv("DMX_BASE_URL", "https://www.dmxapi.com/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    presence_penalty = 0.1,
                )
            elif model == 'qwen2.5-72b-instruct':
                #使用deerapi的api
                model = 'qwen2.5-72b-instruct'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "DMX_API_KEY"),
                            base_url = os.getenv("DMX_BASE_URL", "https://www.dmxapi.com/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    max_tokens=8192,
                    presence_penalty = 0.1,
                )
            elif model == 'gemini-2.5-flash-nothinking':
                #使用deerapi的api
                model = 'gemini-2.5-flash-nothinking'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "DMX_API_KEY"),
                            base_url = os.getenv("DMX_BASE_URL", "https://www.dmxapi.com/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    presence_penalty = 0.1,
                )
            elif model == 'o3':
                #使用deerapi的api
                model = 'o3'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "DEER_API_KEY"),
                            base_url = os.getenv("DEER_BASE_URL", "https://api.deerapi.com/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    presence_penalty = 0.1,
                )
            elif model == 'gpt-4.1-mini':
                #使用deerapi的api
                model = 'gpt-4.1-mini'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "DMX_API_KEY"),
                            base_url = os.getenv("DMX_BASE_URL", "https://www.dmxapi.com/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    presence_penalty = 0.1,
                )
            elif model == 'o4-mini':
                #使用deerapi的api
                model = 'o4-mini'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "DEER_API_KEY"),
                            base_url = os.getenv("DEER_BASE_URL", "https://api.deerapi.com/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    presence_penalty = 0.1,
                )
            elif model == 'claude-3-7-sonnet-thinking':
                #使用deerapi的api
                model = 'claude-3-7-sonnet-thinking'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "DEER_API_KEY"),
                            base_url = os.getenv("DEER_BASE_URL", "https://api.deerapi.com/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    presence_penalty = 0.1,
                )
            elif model == 'gemini-2.5-pro-preview-06-05':
                #使用deerapi的api
                model = 'gemini-2.5-pro-preview-06-05'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "DEER_API_KEY"),
                            base_url = os.getenv("DEER_BASE_URL", "https://api.deerapi.com/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    presence_penalty = 0.1,
                )
            elif model == 'gemini-2.5-flash-preview-05-20':
                #使用deerapi的api
                model = 'gemini-2.5-flash-preview-05-20'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "DEER_API_KEY"),
                            base_url = os.getenv("DEER_BASE_URL", "https://api.deerapi.com/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    presence_penalty = 0.1,
                )
            elif model == 'Pro/google/gemma-2-9b-it':
                #使用deepinfra的api
                model = 'google/gemma-2-9b-it'
                # client = OpenAI(
                #             api_key = _get_api_key(llm_ip, "DEEPINFRA_API_KEY"),
                #             base_url = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai"),#不需要后面的chat/completions,因为会自动加上
                #             )
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "TOGETHER_API_KEY"),
                            base_url = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    presence_penalty = 0.1,
                    max_tokens = 1024,
                )
            elif model == 'Qwen/Qwen2.5-72B-Instruct':
                #使用togrtherai的api
                true_model = 'Qwen/Qwen2.5-72B-Instruct-Turbo'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "TOGETHER_API_KEY"),
                            base_url = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=true_model,
                    presence_penalty = 0.1,
                )
            # elif model == 'Qwen/Qwen2.5-72B-Instruct':
            #     #使用aliyun的api
            #     true_model = 'qwen2.5-72b-instruct'
            #     client = OpenAI(
            #                 api_key = _get_api_key(llm_ip, "DASHSCOPE_API_KEY"),
            #                 base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),#不需要后面的chat/completions,因为会自动加上
            #                 )
            #     completion = client.chat.completions.create(
            #         messages=answer_context,
            #         temperature=temperature,
            #         top_p=top_p,
            #         model=true_model,
            #         presence_penalty = 0.1,
            #     )
            elif model == 'meta-llama/Meta-Llama-3.1-8B-Instruct':
                #使用deepinfra的api
                #model = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
                # client = OpenAI(
                #             api_key = _get_api_key(llm_ip, "DEEPINFRA_API_KEY"),
                #             base_url = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai"),#不需要后面的chat/completions,因为会自动加上
                #             )
                model = 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "TOGETHER_API_KEY"),
                            base_url = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    presence_penalty = 0.1,
                )
            elif model == 'Qwen/Qwen2.5-7B-Instruct-Turbo':
                model = 'Qwen/Qwen2.5-7B-Instruct-Turbo'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "TOGETHER_API_KEY"),
                            base_url = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    presence_penalty = 0.1,
                )
            elif model == 'qwen2.5-14b-instruct':
                model = 'qwen2.5-14b-instruct'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "DMX_API_KEY"),
                            base_url = os.getenv("DMX_BASE_URL", "https://www.dmxapi.com/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    presence_penalty = 0.1,
                )
            elif model == 'DeepSeek-R1-Distill-Qwen-7B':
                model = 'DeepSeek-R1-Distill-Qwen-7B'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "DMX_API_KEY"),
                            base_url = os.getenv("DMX_BASE_URL", "https://www.dmxapi.com/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    presence_penalty = 0.1,
                )
            elif model == 'mistralai/Mixtral-8x7B-Instruct-v0.1':
                #使用togrtherai的api
                true_model = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "TOGETHER_API_KEY"),
                            base_url = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=true_model,
                    presence_penalty = 0.1,
                )

            # elif model == 'meta-llama/Meta-Llama-3.1-70B-Instruct':
            #     #使用aliyun的api
            #     true_model = 'llama3.1-70b-instruct'
            #     client = OpenAI(
            #                 api_key = _get_api_key(llm_ip, "DASHSCOPE_API_KEY"),
            #                 base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),#不需要后面的chat/completions,因为会自动加上
            #                 )
            #     completion = client.chat.completions.create(
            #         messages=answer_context,
            #         temperature=temperature,
            #         top_p=top_p,
            #         model=true_model,
            #         presence_penalty = 0.1,
            #     )
            elif model == 'meta-llama/Meta-Llama-3.1-70B-Instruct':
                #使用togrtherai的api
                true_model = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
                client = OpenAI(
                            api_key = _get_api_key(llm_ip, "TOGETHER_API_KEY"),
                            base_url = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1"),#不需要后面的chat/completions,因为会自动加上
                            )
                completion =  client.chat.completions.create(
                    messages=answer_context,
                    temperature=temperature,
                    top_p=top_p,
                    model=true_model,
                    presence_penalty = 0.1,
                )
            else:
                client = OpenAI(
                    api_key = _get_api_key(llm_ip, "TOGETHER_API_KEY"),
                            base_url = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1"),
                    )

                completion =  client.chat.completions.create(
                    messages= answer_context,
                    temperature= temperature,
                    top_p= top_p,
                    model= model,
                    presence_penalty = 0.1,
                    )
                #试一下用siliconcloud的api
                # client = OpenAI(
                #     api_key = _get_api_key(llm_ip, "SILICONFLOW_API_KEY"),
                #     base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
                #     )

                # completion = client.chat.completions.create(
                #     messages= answer_context,
                #     temperature= temperature,
                #     top_p= top_p,
                #     model= model,
                #     presence_penalty = 0.1,
                #     )
            #completion["usage"] = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        else: # OpenAI GPT
            #使用deerapi的api
            client = OpenAI(
                        api_key = _get_api_key(llm_ip, "DMX_API_KEY"),
                        base_url = os.getenv("DMX_BASE_URL", "https://www.dmxapi.com/v1"),#不需要后面的chat/completions,因为会自动加上
                        )
            completion =  client.chat.completions.create(
                messages=answer_context,
                temperature=temperature,
                top_p=top_p,
                model=model,
                presence_penalty = 0.1,
            )
    except Exception as e:
        print(e, flush=True)
        if 'Content Exists Risk' in str(e):
            print("Risk detected, change model", flush=True)
            time.sleep(2)
            return generate_answer(answer_context, 'gemini-2.5-flash-nothinking', llm_ip, temperature=temperature, top_p=top_p)
        if 'maximum' in str(e) or 'max_output_token' in str(e):
            print("max token limit exceeded, retrying......", flush=True)
        print("retrying due to an error......", flush=True)
        time.sleep(5)
        return generate_answer(answer_context, model, llm_ip, temperature=temperature, top_p=top_p)
    #print(completion)
    return completion
