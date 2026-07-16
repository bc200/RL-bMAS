

BLACKBOARD_TEMPLATE_NO_HIS = """
You are the central controller of a multi-agent system. Your goal is to schedule the most appropriate agent to solve the problem based on the current progress. Each agent corresponds to one action. The agent names and their corresponding descriptions are listed below:\n {role_list}.\n
The given problem is:{question}. The execution history of the agent is recorded on the blackboard.
you have already taken 0 step(s). Currently the blackboard is empty.
you should choose an admissible agent for current step. Consider which agent can contribute new information or resolve current uncertainty. Ensure correct and efficient problem-solving.
Action agents may have different capabilities from yours, so you should consider diverse problem-solving strategies to fully utilize the available agents.
Only select the most appropriate agent for the current step, do not solve the problem yourself.
# Please:
# 1. Briefly reason which type of agent is most suitable at this step in 2-3 sentences. The reasoning process are enclosed within <thought> </thought> tags.
# 2. Output the chosen agent within <action> </action> tags. e.g., "<thought> reasoning process here </thought>
<action> terminate </action>".
"""

BLACKBOARD_TEMPLATE = """
Your task is to schedule several agents to cooperate and solve the given problem. Each agent corresponds to one action. The agent actions and their corresponding descriptions are listed below:\n {role_list}.\n
The given problem is:{question}. The execution history of the agent is recorded on the blackboard. The blackboard contains discussions (the execution history of agents) and tasks (tasks to be solved).
Prior to this step, you have already taken {step_count} step(s). In the previous step you choose {previous_action}, agent choice history: {action_history}\n You are now at step {current_step} and current blackboard content is: {current_blackboard}.
you should choose an admissible agent for current step. Consider which agent can contribute new information or resolve current uncertainty. Ensure correct and efficient problem-solving.
Action agents may have different capabilities from yours, so you should consider diverse problem-solving strategies to fully utilize the available agents.
Only select the most appropriate agent for the current step, do not solve the problem yourself.
# Please:
# 1. Briefly reason which type of agent is most suitable at this step in 2-3 sentences. The reasoning process are enclosed within <thought> </thought> tags.
# 2. Output the chosen agent within <action> </action> tags. e.g., "<thought> reasoning process here </thought>
<action> terminate </action>".
"""
# Do not output anything else.
#you should choose an admissible agent for current step. Consider which agent can contribute new information or resolve current uncertainty. Present your choice within <action> </action> tags. For example if you want to choose decide agent then you should output "<action> decide </action>". 
#When there is too much content on the blackboard, it needs to be cleaned up by "clean" or "summary" agent, otherwise it will slow down the system efficiency.
# Additional guidance:
# Action agents may have different capabilities from yours, so you should consider diverse problem-solving strategies to fully utilize the available agents.