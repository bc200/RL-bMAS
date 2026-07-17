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

# This file contains the Agent class, which is the main class for the agent. The agent is responsible for interacting with the model and processing the output.

import os

from prompt_lib import init_role_prompt, role_blackboard_prompt
from utils import generate_answer


problem_prompt = "The original problem is:"
#bb_prompt = "The messages currently on the blackboard are listed below:\n "#TODO 完善prompt
def construct_message(state, self_role, other_prompt=""):
    messages = ""
    #messages += other_prompt#TODO
    #messages += bb_prompt
    #messages += str(state).replace(self_role, "assistant(you)") + "\n"
    messages += other_prompt#.replace("{agent_role}", "assistant(you)")

    #messages += other_prompt
    return {"role": "user", "content": messages}#TODO:能否不使用user

def construct_message_empty(state, other_prompt=""):
    messages = ""
    #messages += other_prompt#TODO
    #messages += " Currently the blackboard are empty."
    #messages += "\n"
    messages += other_prompt
    return {"role": "user", "content": messages}#TODO:能否不使用user


class Agent:
    def __init__(self, role, mtype, temperature=1, top_p=1, api_key = None):
        """Create an agent

        Args:
            role (str): name of this agent
            mtype (str): model name
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
        """
        self.role = role
        self.mtype = mtype
        self.reply = None
        self.answer = ""
        self.question = None
        self.llm_ip = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.temperature = temperature
        self.top_p = top_p
        self.probability = 0
        self.memory = []
        #self.init_prompt = init_prompt
        
        #self.set_meta_prompt(self.init_prompt)
        if mtype == "gpt-35-turbo":
            self.model = "gpt-35-turbo"
        elif mtype == "gpt-4":
            self.model = "gpt-4"
        elif mtype == "gpt-4-1106-Preview":
            self.model = "gpt-4-1106-Preview"
        elif mtype == "gpt-35-turbo-1106":
            self.model = "gpt-35-turbo-1106"
        else:
            self.model = mtype
        self.llm_ip = api_key
        

    def get_reply(self):
        return self.reply

    def get_answer(self, input_context):
        self.question = input_context
        self.memory.append(input_context)
        #print('Memory:', self.memory)
        output_context = generate_answer(self.memory, self.model, self.llm_ip, self.temperature, self.top_p)
        #print(output_context)
        #删除memory中的input_context
        self.memory.pop()
        #self.memory.append({"role": "assistant", "content": output_context.choices[0].message.content})
        prompt_tokens = output_context.usage.prompt_tokens
        completion_tokens = output_context.usage.completion_tokens
        return output_context.choices[0].message.content, prompt_tokens, completion_tokens

    def preprocess(self, state, other_prompt=""):
        contexts = []
        #contexts.append(self.set_meta_prompt(self.init_prompt))########这个是错误的吧
        contexts.append(construct_message(state, self.role, other_prompt))
        return contexts
    
    def preprocess_empty(self, state, other_prompt=""):#为了应对bb空的时候小模型无的controller无法选择的情况
        contexts = []
        #contexts.append(self.set_meta_prompt(self.init_prompt))########这个是错误的吧
        contexts.append(construct_message_empty(state, other_prompt))
        return contexts

    # def postprocess(self, completion, question):
    #     self.reply = completion
    #     self.answer, _ = self.ans_parser(self.reply, question)

    def set_meta_prompt(self, prompt):
        """Set the meta_prompt

        """
        meta_prompt = f"{prompt}"
        self.memory = []
        self.memory.append({"role": "system", "content": meta_prompt})
        return {"role": "system", "content": meta_prompt}
    
    #需要memory功能让智能体记住自己之前说的话(这里是问题)
    def get_memory(self):
        return self.memory

reason_prompt = """
Based on the above facts and assumptions, please conduct logical reasoning step by step and provide the most reasonable conclusion. Avoid repetitive explanations.
"""

class ReasonAgent(Agent):
    def __init__(self, role, mtype, temperature=1, top_p=1, api_key = None):
        super().__init__(role, mtype, temperature, top_p, api_key)

    def get_answer(self, input_context):
        self.question = input_context
        self.memory.append(input_context)
        #print('Memory:', self.memory)
        step1_output = generate_answer(self.memory, self.model, self.llm_ip, self.temperature, self.top_p)
        step1_context = step1_output.choices[0].message.content
        prompt_step1 = step1_output.usage.prompt_tokens
        completion_step1 = step1_output.usage.completion_tokens
        self.memory.append({"role": "assistant", "content": step1_context})
        self.memory.append({"role": "user", "content": reason_prompt})
        step2_output = generate_answer(self.memory, self.model, self.llm_ip, self.temperature, self.top_p)
        step2_context = step2_output.choices[0].message.content
        prompt_step2 = step2_output.usage.prompt_tokens
        completion_step2 = step2_output.usage.completion_tokens
        #print('##############################reasoning memory:', self.memory,'\n',step2_context)
        self.memory = [self.memory[0]]#只保留system prompt
        return '<output>'+step2_context+'</output>', prompt_step1+prompt_step2, completion_step1+completion_step2

import json
import requests

REACT_PROMPT = """
You are a helpful and efficient research assistant.

Your task is to analyze the user's input text below and decide whether you need to perform a Google search to find the latest information. You will follow these steps:
1.  Carefully read the text and identify all questions, entities, or topics that require external, up-to-date information from the internet.
2.  For each identified item, use the 'Google Search_serper' tool to find relevant information.
3.  Synthesize the search results into a clear and concise summary to answer the user's implicit or explicit questions. Do not answer from your own knowledge. Base your final answer ONLY on the information you find using the search tool.

If you decide to use the search tool, format your response as follows:
```json
{
    "action": "Google_Search_serper",
    "action_query": "a list of questions or topics to search for"
}
```
If you do not need to search, simply respond with:
```json
{
    "action": "None",
    "action_query": ""
}
```
Once you have the search results, summarize them in a way that directly addresses the user's questions or needs. format your response as follows:
```json
{
    "action": "Summary",
    "summary": "Your concise summary of the search results."
}
```

Begin!
"""

class ToolAgent(Agent):
    def __init__(self, role, mtype, temperature=1, top_p=1, api_key = None):
        super().__init__(role, mtype, temperature, top_p, api_key)

    def Google_Search_serper(self, query: str) -> str:
        """使用Google Serper API进行在线搜索并返回结果摘要。"""
        #print(f"--- 正在执行工具 [Google_Search_serper] 查询: {query} ---")
        serper_api_key = os.getenv("SERPER_API_KEY")
        if not serper_api_key:
            return "SERPER_API_KEY is not configured."
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            response.raise_for_status() # 如果请求失败则抛出异常
            results = response.json()
            
            # 提取并简化搜索结果用于返回给LLM
            if "organic" in results and len(results["organic"]) > 0:
                snippets = [
                    f"Title: {item.get('title', '')}\nSnippet: {item.get('snippet', '')}"
                    for item in results["organic"][:3] # 取前3个结果
                ]
                return "\n\n".join(snippets)
            elif "answerBox" in results:
                return results["answerBox"].get("snippet") or results["answerBox"].get("title", "No snippet found.")
            else:
                return "No relevant search results found."
        except Exception as e:
            print(f"工具执行出错: {e}")
            return f"Error performing search: {e}"
        
    def get_answer(self, input_context):
        self.question = input_context
        self.memory.append(input_context)
        max_turns = 5  # 限制最大轮次，避免无限循环
        prompt_tokens_total = 0
        completion_tokens_total = 0
        for turn in range(max_turns):
            #print('memory:', self.memory)
            llm_output = generate_answer(self.memory, self.model, self.llm_ip, self.temperature, self.top_p)
            output_context = llm_output.choices[0].message.content
            prompt_tokens_total += llm_output.usage.prompt_tokens
            completion_tokens_total += llm_output.usage.completion_tokens
            #print(f"--- LLM 回合 {turn+1} 原始输出 ---\n{output_context}")
            self.memory.append({"role": "assistant", "content": output_context})
            #print(f"--- LLM 回合 {turn+1} 输出 ---\n{output_context}")
            try:
                # 找到JSON块的开始和结束位置
                json_start = output_context.find('{')
                json_end = output_context.rfind('}') + 1
                action_json_str = output_context[json_start:json_end]
                action_data = json.loads(action_json_str)
                #print(f"--- LLM 回合 {turn+1} 输出 ---LLM Action:\n{action_data.get("action")}")
            except (json.JSONDecodeError, IndexError) as e:
                print(f"错误：无法解析LLM的行动指令。错误: {e}")
                #重新采样
                self.memory.pop()  # 移除无法解析的输出
                continue
            if action_data.get("action") is None:
                print("LLM的行动指令为空，重新采样。",self.memory)
            if action_data.get("action").strip().lower() == "stop":
                print("LLM认为不需要执行任何行动。")
                self.memory = [self.memory[0]]#只保留system prompt
                return "<output>{no need for websearching, waiting for more information}</output>", prompt_tokens_total, completion_tokens_total
            elif action_data.get("action").strip().lower() == "google_search_serper":
                # 3. 执行行动
                action_query = action_data.get("action_query")
                action_list = action_query
                # 调用工具函数进行搜索
                observations = []
                for temp_action in action_list:
                    observation = self.Google_Search_serper(temp_action)
                    observations.append(observation)
                #print(f"--- 工具观察结果 ---\n{observation}")
                # 将观察结果添加到历史记录，供LLM下一轮思考
                self.memory.append({"role": "user", "content": f"Observation:\n{observations}"})
            elif action_data.get("action").strip().lower() == "summary":
                summary = action_data.get("summary", "")
                #print(f"LLM总结结果:\n{summary}")
                self.memory = [self.memory[0]]#只保留system prompt
                return '<output>{'+str(summary)+'}</output>', prompt_tokens_total, completion_tokens_total
        self.memory = [self.memory[0]]#只保留system prompt
        return "<output>{no need for websearching, waiting for more information}</output>", prompt_tokens_total, completion_tokens_total

from typing import List, Union
from langchain_openai import ChatOpenAI
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, GoogleSerperAPIWrapper
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent#, AgentExecutor
from langchain_classic.agents import AgentExecutor
#from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler

from langgraph.errors import GraphRecursionError
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import requests
import time

API_KEY = os.getenv("CUSTOM_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("CUSTOM_BASE_URL", "https://www.dmxapi.com/v1") # 示例地址

# 全局初始化一次 embedding 模型，避免重复加载
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=API_KEY, base_url=BASE_URL)

# ==========================================
# 定义工具 (Tools)
# ==========================================
# --- Agent 1 工具: 科学检索 ---
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
pubmed = PubmedQueryRun()

# --- Agent 2 工具: 计算 ---
repl = PythonREPL()

import wikipedia
# ==================== 代理 + User-Agent 配置 ====================
PROXY_URL = os.getenv("WIKIMEDIA_PROXY_URL")
# User-Agent 遵循 Wikipedia 规范: name/version (contact)
USER_AGENT = os.getenv("WIKIMEDIA_USER_AGENT", "verl-agent/1.0")
# Patch wikipedia 库的底层 session
_wiki_session = requests.Session()
_wiki_session.headers.update({"User-Agent": USER_AGENT})
if PROXY_URL:
    _wiki_session.proxies = {
        "http": PROXY_URL,
        "https": PROXY_URL,
    }
# 覆盖 wikipedia 库的默认 session
wikipedia.session = _wiki_session  # type: ignore
# ==================== 初始化 Wrapper ====================
wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=3000
)
# ==================== Tool 定义 ====================
@tool
def robust_wikipedia_tool(query: str) -> str:
    """
    Search Wikipedia for academic definitions, historical events, or scientific concepts.
    Use this when you need a verified encyclopedia definition.
    """
    clean_query = query.strip().replace("\n", " ")[:200]
    for attempt in range(3):
        try:
            result = wiki_wrapper.run(clean_query)
            if result:
                return result
            return "No Wikipedia content found for this query."
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ Wikipedia 失败 [{attempt+1}/3]: {error_msg}")
            print(f"   Query: {clean_query}")
            if attempt < 2:
                wait = min(2 ** attempt,30)  # 指数退避，最大30秒
                print(f"   等待 {wait}s 重试...")
                time.sleep(wait)
            else:
                return (
                    f"Wikipedia search failed after 3 attempts. "
                    f"Error: {error_msg}. "
                    f"Please use Google Search instead."
                )

# wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=3000)
# @tool
# def robust_wikipedia_tool(query: str):
#     """
#     Search Wikipedia for academic definitions, historical events, or scientific concepts.
#     Use this when you need a verified encyclopedia definition.
#     """
#     try:
#         # 尝试调用
#         result = wiki_wrapper.run(query)
#         # 如果返回空字符串，说明没查到
#         if not result:
#             return "No content found on Wikipedia for this query."
#         return result
#     except Exception as e:
#         # 核心修改：捕获所有错误（包括 JSONDecodeError），返回错误字符串而不是抛出异常
#         print(f"⚠️ Wikipedia API 调用失败: {e}")
#         print('query:', query, 'Please check if the query is valid and try again.')
#         return f"Wikipedia search failed due to network or parsing error: {str(e)}. Please try using Google Search tool instead."


# @tool
# def python_calculator(code: str):
#     """
#     Execute Python code for mathematical calculations and programming.
#     The input must be a pure Python code string.
#     The output of print() will be captured.
#     """
#     try:
#         #print(f"Executing Python code: {code}")
#         result = repl.run(code)
#         #print(f"code execution result: {result}")
#         return f"Code execution result:\n{result}"
#     except Exception as e:
#         return f"Code execution result: {e}"
import subprocess
@tool
def python_calculator(code: str):
    """
    Execute Python code for mathematical calculations and programming.
    """
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=60  # 30秒超时，超时直接杀进程
        )
        output = result.stdout.strip()
        error = result.stderr.strip()
        if result.returncode != 0:
            return f"Code execution error:\n{error}"
        return f"Code execution result:\n{output}" if output else f"Code executed (no stdout). stderr: {error}"
    except subprocess.TimeoutExpired:
        return "Code execution timed out after 30s. Possible infinite loop."
    except Exception as e:
        return f"Code execution failed: {str(e)}"

# --- Agent 3 工具: 搜索 ---
# 确保环境变量 SERPER_API_KEY 已设置
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY) if SERPER_API_KEY else None

@tool
def google_search(query: str):
    """Use Google to search for real-time information."""
    if search is None:
        return "SERPER_API_KEY is not configured."
    # 这里我们不用 search.run(query)，因为那个只返回摘要
    # 我们用 search.results(query)，它会返回完整的 JSON 数据
    try:
        # -------------------------------------------------------
        # 修改点：增加异常捕获，防止 Serper 返回空导致程序崩溃
        # -------------------------------------------------------
        results = search.results(query)
    except Exception as e:
        # 如果 API 调用失败（比如网络问题或 Key 问题），返回错误信息给 LLM
        # LLM 看到错误后，通常会尝试换个问法或者停止
        print(f"❌ Serper API 报错: {e}")
        return f"Error: Google Search API failed to return results. Details: {str(e)}"
    # 我们来处理一下数据，把前 5 个结果提取出来
    output = ""
    # 增加非空判断
    if not results or not isinstance(results, dict):
        return "No results found or API returned invalid format."
    if "organic" in results:
        for item in results["organic"][:5]: # 只看前5条，省点眼力
            title = item.get("title", "NULL")
            link = item.get("link", "NULL")
            snippet = item.get("snippet", "NULL")

            # 把这些信息拼成一段话给大模型看
            output += f"Title: {title}\n URL Link: {link}\nSnippet: {snippet}\n\n"
    return output
@tool
def visit_page_tool(url: str, query: str) -> str:
    """
    Visit a webpage and extract content RELEVANT to the query.
    
    Args:
        url: The http link to visit.
        query: The specific question or keywords to look for within the page content.
    """
    print(f"\n👴 正在读取: {url}")
    full_text = ""  # ✅ 必须先初始化！原代码没有这行会报 UnboundLocalError
    # --- 第一招：尝试 Jina (记得去申请个免费Key效果更好，没有也能跑) ---
    try:
        # 这里建议填入您的 Jina API Key，如果没有就留空，但容易遇到您刚才的报错
        # jina_headers = {
        #     "Authorization": "Bearer jina_..." 
        # }
        # 如果没有Key，就把 headers=jina_headers 去掉
        response = requests.get(f"https://r.jina.ai/{url}", timeout=10)
        
        if response.status_code == 200 and "SecurityCompromiseError" not in response.text:
            full_text = response.text
    except:
        print("Jina 读取失败，切换到本地模式...")

    # --- 第二招：本地读取 (Trafilatura) ---
    # 这是您的保底手段，不依赖外部服务
    import trafilatura
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            full_text = trafilatura.extract(downloaded)
        else:
            return "url could not be fetched."+"\n\nSystem Note: Access failed. Please consider summarize based on the Google Search Snippets you already have or try another link."
            
    except Exception as e:
        return f"url could not be fetched: {str(e)}"+"\n\nSystem Note: Access failed. Please consider summarize based on the Google Search Snippets you already have or try another link."

    if not full_text: return "Empty page."

    # 1. 切分文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    docs = text_splitter.create_documents([full_text])
    
    if len(docs) == 0 or len(full_text)<3000 : return full_text[:3000]

    try:
        # 2. 建立临时向量索引 (内存中，速度很快)
        db = FAISS.from_documents(docs, embeddings)
        
        # 3. 语义搜索最相关的 3 个片段
        relevant_docs = db.similarity_search(query, k=3)
        
        # 4. 拼接结果
        result_content = "\n\n".join([d.page_content for d in relevant_docs])
        #print(f"👴 读取到相关内容: {result_content} ")
        return f"Relevant content found:\n{result_content}"
        
    except Exception as e:
        print(f"Vector search failed: {e}, falling back to head text.")
        return full_text[:3000]
# 定义一个监控器类
class GrandpaToolMonitor(BaseCallbackHandler):
    """
    爷爷专属监控器：只关心工具的使用情况，不打印别的废话。
    """
    
    # 当智能体【开始】使用工具时触发
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name")
        #print(f"\n🔧 [监控] 正在调用工具: 【{tool_name}】")
        #print(f"   📥 输入参数: {input_str}")

    # 当工具【结束】运行并返回结果时触发
    def on_tool_end(self, output, **kwargs):
        # 截取一下，防止网页内容太长刷屏
        short_output = str(output)[:500] + "..." if len(str(output)) > 500 else output
        #print(f"   ✅ 工具返回结果: {short_output}\n")

    # 当工具【报错】时触发
    def on_tool_error(self, error, **kwargs):
        print(f"   ❌ 工具出错了: {error}\n")

# --- 使用方法 ---

# 1. 实例化监控器
my_monitor = GrandpaToolMonitor()

class SpecializedToolAgent(Agent):
    def __init__(self, role, mtype, temperature=0, top_p=1, api_key=None, 
                 ):
        """Create a specialized agent that uses a specific tool.
        """
        super().__init__(role, mtype, temperature, top_p, api_key)
        
        # 1. 设置环境变量 (LangChain 依赖)
        #os.environ["OPENAI_API_KEY"] = api_key or ""

        # 2. 初始化 LLM
        self.llm = ChatOpenAI(model=mtype,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.3, # 保持低温以获得稳定的工具调用
        max_retries=3,  # 【关键配置】遇到网络错误或 50x 错误时自动重试 3 次
        request_timeout=60 # 也可以适当增加超时时间
        )

        self.tools = self._get_single_tool(role)

        system_msg = self.define_system_msg(role)
        

        # 5. 创建 Agent Executor
        self.agent_structure = create_agent(self.llm, self.tools, system_prompt=system_msg)
        # self.agent_executor = AgentExecutor(
        #     agent=self.agent_structure, 
        #     tools=self.tools, 
        #     verbose=True, # 调试时设为True，可以看到是否调用了工具
        #     handle_parsing_errors=True 
        # )
        self.agent_executor = self.agent_structure

    def _get_single_tool(self, tool_name):
        """工厂方法：根据名称返回具体的 LangChain Tool 对象"""
        name_lower = tool_name.lower()
        
        if "web" in name_lower or "serper" in name_lower:
            return [google_search, visit_page_tool]

        elif "retrieve" in name_lower:
            return [robust_wikipedia_tool, pubmed]
            
        # elif "wolfram" in name_lower or "math" in name_lower:
        #     return WolframAlphaQueryRun(api_wrapper=WolframAlphaAPIWrapper())
            
        elif "python" in name_lower or "code" in name_lower:
            return [python_calculator]
            
        else:
            raise ValueError(f"Unknown tool name: {tool_name}. Supported: google, wiki, python")

#     def define_system_msg(self, role):
#         """定义系统消息"""
#         role_lower = role.lower()

#         if 'web' in role_lower or 'search' in role_lower:
#             return f"""You are web researcher. You focus on real-world information search on the Internet. You have access to the following tools: {self.tools}.
# ### ⚠️ CRITICAL TOOL USAGE INSTRUCTIONS:
# 1. **`google_search`**: 
#    - Use this to find information sources. 

# 2. **`visit_page_tool(url, query)`**: 
#    - **UPDATE**: This tool now requires TWO arguments:
#      - `url`: The link you want to read.
#      - `query`: The specific fact or topic you are looking for on that page (e.g., "iPhone 15 battery life" or "Elon Musk birth date").
#    - **Why?**: The tool uses your `query` to scan the webpage and only returns the relevant paragraphs. A good `query` ensures you get the right answer instead of useless headers/footers.

# ###  Your Task:
# 1. Read the user provided text context carefully and identify important questions, entities, or topics that require external, up-to-date information from the internet. Construct the appropriate query for the google search tool and execute searching.
# 2. **Carefully** read the summary and title of the search results. Check search results. Can you answer now? -> If YES, Answer.
# 3. If NO, pick the best link, visit it, read the content, synthesize the search results and webpage content into a clear and concise summary.".
# ### STRICT RULES:
# 1. **Limit Searches**: use no more than 3 google_search query per user request. Focus on core questions.
# 2. **Analyze Snippets First**: After searching, read the 'Snippet' in the search results carefully. If the answer is already in the snippet, synthesize the answer immediately using that info. DO NOT visit the page. Use `visit_page_tool` ONLY if the snippet is missing critical details. 
# 3. **Single Click Policy**: Usually, clicking the ONE most relevant link is enough. Do not click multiple links.
# 4. **Stop Condition**: Once you have gathered enough information from search snippet or visited page, immediately provide your final answer. Do not loop back to search again.

# Do not fabricate information. Only answer based on the tool's output.
#             """
#         elif 'python' in role_lower or 'code' in role_lower:
#             return f"""You are python executor. You focus on writing Python code to solve mathematical calculations and programming problems. You have access to the following tools: {self.tools}.
#             Your Task:
# 1. Read the user provided text context carefully.
# 2. Determine if it is necessary to execute any code or perform calculations.
# 3. If YES: Construct the appropriate code snippet, execute it, and synthesize the results into a clear and concise summary.
# 4. If NO (the context is irrelevant to your tool or fully answered): Reply with "No action needed".

# Do not fabricate information. Only answer based on the tool's output.
#             """
#         elif 'retrieve' in role_lower:
#             return f"""You are knowledge retriever. You focus on retrieving knowledge, including common sense knowledge, scientific knowledge, academic definitions, and medical papers. You have access to the following tools: {self.tools}.
#             Your Task:
# 1. Read the user provided text context carefully.
# 2. Determine if it is necessary to perform any scientific research on wikipedia and pubmed.
# 3. If YES: Construct the appropriate query for the wikipedia and pubmed search, execute it, and synthesize the results into a clear and concise summary.
# 4. If NO (the context is irrelevant to your tool or fully answered): Reply with "No action needed".

# Do not fabricate information. Only answer based on the tool's output.
#             """
#         else:
#             print("No specialized tool found for this role.")
#         return f"You are a specialized agent named {self.role}. You have access to the following tools: {self.tools}."

    def define_system_msg(self, role):
        role_lower = role.lower()
        
        # 基础工具描述
        base_tools = f"You have access to the following tools: {self.tools}."

        if 'web' in role_lower or 'search' in role_lower:
            return f"""You are an advanced Web Researcher Agent.
{base_tools}

### 🛠️ TOOL USAGE PROTOCOL (STRICT):
1. **`google_search`**: Use simple keywords. 
2. **`visit_page_tool(url, query)`**: 
   - **MANDATORY**: You MUST provide TWO arguments: `url` and `query`.
   - `query`: A specific question to extract from the page (e.g., "iPhone 15 battery specs").
   - **Strategy**: Only visit a page if the search snippet is insufficient.

### 🧠 WORKFLOW (Follow these steps):
1. **Analyze**: Read the entire input context, identify one most important query to search for a piece of relevant and missing information.
2. **Search**: Execute `google_search`.
3. **Evaluate Snippets**: 
   - Read the search results carefully. 
   - **CRITICAL**: If the answer is in the snippet/summary, **STOP** and answer immediately. Do NOT visit the page just to "confirm".
4. **Visit (Optional)**: ONLY if snippets are missing data, pick the **top 1** most relevant link and use `visit_page_tool`. 
5. **Finalize**: Based on the information you have gathered so far above, objectively summarize the information. Do not add any personal interpretation or assumptions.

### 🚫 CONSTRAINTS:
- **Max limit**: Max 3 searches. Max 1 page visit per search.
- **No Looping**: Once you have a plausible answer, provide it. Do not keep searching for "better" phrasing.
- If the answer cannot be found, state "Information not found" and stop.
"""

        elif 'python' in role_lower or 'code' in role_lower:
            return f"""You are a Python Executor Agent.
{base_tools}

### CRITICAL CONTEXT HANDLING:
The user input may contain question and relevant discussions.
1. **Parse**: Extract all relevant numbers, data structures (lists, dictionaries), or mathematical formulas from the text.
2. **Clarify**: If the text contains ambiguity, make a reasonable assumption and state it in your code comments.

### WORKFLOW:
1. **Plan**: Think about the algorithm needed to solve the problem.
2. **Code**: Write **complete, executable Python code**.
   - Import necessary libraries (math, datetime, etc.).
   - **PRINT** the final result to `stdout`. The tool only captures printed output.
3. **Verify**: 
   - Look at the code output. Does it make sense? 
   - If the code errors out, analyze the error message, **FIX** the code, and run it again.
   - You are allowed to retry code execution to fix bugs.

### STRICT RULES:
- **No Partial Code**: Always write the full script needed to get the answer.
- **Output**: Ensure your code prints the answer clearly (e.g., `print(f"The answer is {{result}}")`).
- **Safety**: Do not write code that accesses the internet or deletes files.
"""

        elif 'retrieve' in role_lower:
            return f"""You are a Knowledge Retriever Agent.
{base_tools}

### CRITICAL CONTEXT HANDLING:
The user input may contain question and relevant discussions. You must distinguish between "Common Knowledge" (which you might already know) and "Specific Facts" (which need retrieval).
- Only use tools for definitions, medical facts, scientific data, or obscure historical events.

### WORKFLOW:
1. **Decompose**: If the user asks a complex question (e.g., "Compare the GDP of A and B"), break it into simpler sub-questions ("What is the GDP of A?" and "What is the GDP of B?").
2. **Search**: Use `wikipedia` or `pubmed`. Each query should be concise and specific.
3. **Refine**: If the first query returns "No results", try a broader synonym (e.g., change "Covid-19 symptoms in 2024" to "Coronavirus symptoms").
4. **Synthesize**: Combine the retrieved fragments into a coherent summary.

### STRICT RULES:
- **Relevance**: Do not retrieve long articles if a short definition suffices.
- **Honesty**: If the database has no info, state it clearly.
"""

        else:
            return f"You are a specialized agent named {self.role}. {base_tools}"

    def _convert_msg_to_langchain(self, msg_list: List[dict]):
        """将 msg 转换为 LangChain 消息格式"""
        history = []
        for msg in msg_list:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history.append(AIMessage(content=msg["content"]))
        return history

    def get_answer(self, input_context):
        """
        执行流程：
        1. 判断输入是否需要工具 -> 
        2. 如果需要，LLM 生成参数调用工具 -> 获取结果 -> LLM生成总结。
        3. 如果不需要，直接返回回复。
        """
        """
        Args:
            input_context ：字典
        """
        # start LangChain execution
        # 这里不需要传入 chat_history，因为我们假设它是无状态的 Worker
        inputs = {"messages": self._convert_msg_to_langchain([input_context])}
        # 用于记录最新的对话历史（包含工具调用的结果）
        latest_messages = inputs["messages"]
        
        with get_openai_callback() as cb:
            try:
                # -----------------------------------------------------------
                # 关键修改：使用 stream 而不是 invoke
                # stream_mode="values" 会返回每一步更新后的完整 messages 列表
                # -----------------------------------------------------------
                events = self.agent_executor.stream(
                    inputs, 
                    config={
                        "recursion_limit": 10, 
                        "callbacks": [my_monitor]
                    },
                    stream_mode="values" 
                )
                
                # 遍历事件，不断更新 latest_messages
                for event in events:
                    # event["messages"] 是当前的完整对话历史
                    if "messages" in event:
                        latest_messages = event["messages"]
                
                # 如果正常结束，最后一条消息就是结果
                final_output = latest_messages[-1].content
                #print('正常结束，output:\n', final_output)

            except GraphRecursionError:
                import time
                #print(f"\n⚠️ 警告：已达到最大轮数限制 ({10})，正在强制总结...")
                
                # -----------------------------------------------------------
                # 这里的逻辑对应旧版的 early_stopping_method="generate"
                # -----------------------------------------------------------
                
                # 1. 构造一个提示，告诉模型停止搜索，根据现有信息回答
                fallback_prompt = [
                    *latest_messages, # 之前所有的工具调用结果都在这里
                    HumanMessage(content="System Notification: You have reached the maximum number of steps (time limit). Please STOP using any tools. Based on the information you have gathered so far above, objectively summarize the information. Do not add any personal interpretation or assumptions.")
                ]
                # -----------------------------------------------------------
                # 关键修改：手动增加网络错误重试机制
                # -----------------------------------------------------------
                max_retries = 5         # 最大重试次数（你可以根据需要修改）
                retry_delay = 15        # 每次重试等待的时间（秒），给VPN切换节点留出时间
                fallback_success = False


                # 2. 直接调用 LLM (self.llm) 而不是 Agent，生成最终回复
                for attempt in range(max_retries):
                    try:
                        # print(f"🔄 正在尝试生成总结 (第 {attempt + 1}/{max_retries} 次)...")
                        # 2. 调用 LLM
                        fallback_response = self.llm.invoke(fallback_prompt)
                        final_output = f"[Stopped due to limit] {fallback_response.content}"
                        fallback_success = True
                        break # 如果成功，立刻跳出重试循环
                        
                    except Exception as llm_error:
                        print(f"⚠️ 第 {attempt + 1} 次总结失败，遇到网络/API错误: {llm_error}")
                        if attempt < max_retries - 1:
                            print(f"⏳ 等待 {retry_delay} 秒后进行下一次重试...")
                            time.sleep(retry_delay) # 暂停程序，等待网络恢复
                        else:
                            print("❌ 已达到最大重试次数，放弃生成总结。")
                
                # 3. 如果所有重试都失败了，给一个保底的输出，坚决不让程序崩溃
                if not fallback_success:
                    final_output = "[Stopped due to limit] Tool execution failed due to persistent network errors."
                #print(f"\n⏹️ 已强制停止工具调用，生成的总结:\n{final_output}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"发生其他错误: {e}")
                final_output = f"Tool execution failed: {str(e)}"

                # invoke 触发 LLM 思考 -> 工具选择 -> 执行 -> 总结
                
            #     #print('--- SpecializedToolAgent 输入 ---\n', inputs)
            #     response = self.agent_executor.invoke(inputs, config={"recursion_limit": 20, 'callbacks': [my_monitor]})
            #     #print('--- SpecializedToolAgent 输出 ---\n', response)
            #     final_output = response["messages"][-1].content
            # except Exception as e:
            #     print(f"一般都是超过轮数: {e}")
            #     final_output = f"Tool execution failed: {str(e)}"

        # 统计 Token (Langchain callback 自动捕获)
        p_tokens = cb.prompt_tokens
        c_tokens = cb.completion_tokens
        print(f"SpecializedToolAgent {self.role} Token 统计 - Prompt: {p_tokens}, Completion: {c_tokens}")
        # 格式化输出
        # 如果 LLM 判定不需要工具，它会按照 System Prompt 输出 "No action needed" 之类的
        return f"<output>{{{final_output}}}</output>", p_tokens, c_tokens
