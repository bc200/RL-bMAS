import re
from typing import List

def blackboard_projection(actions: List[str]):
    """
    将 LLM 的原始文本输出映射到环境可以理解的离散动作索引。
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format:
        <action>expert/decompose/clean/summary/tool/critic</action>
     

    这个函数会尝试两种策略来解析动作：
    1. 优先尝试从文本中直接提取数字。
    2. 如果没有找到数字，则尝试根据智能体的名字进行匹配。

    :param llm_response: LLM 生成的原始字符串。
    :param agent_names: 有效的智能体名称列表，用于名称匹配。
    :return: 解析出的智能体索引。如果无法解析，则返回 -1 表示无效动作。
    """
    # 策略 1: 尝试从字符串中提取第一个出现的数字
    # 这对于模型输出 "选择 0" 或 "action: 0" 等格式非常有效

    valids = [0] * len(actions)
    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                print('no action signal, output:',actions[i])
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-20:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            # if extracted_action.lower().strip() == 'verify':
            #     print('我倒要看看为啥选verify', actions[i])
            
            actions[i] = extracted_action
            valids[i] = 1

        except:
            # randomly choose an action from the action list if illegal
            actions[i] = actions[i][-20:]

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids
