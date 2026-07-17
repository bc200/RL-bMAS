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

"""
Test the text gym environment.

TODO: move to testing dir for more rigorous tests
"""
import gym
from rich import print
from rich.markup import escape

from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.models import RandomPolicy
from web_agent_site.utils import DEBUG_PROD_SIZE

if __name__ == '__main__':
    env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=DEBUG_PROD_SIZE)
    env.reset()
    
    try:
        policy = RandomPolicy()
    
        observation = env.observation
        while True:
            print(observation)
            available_actions = env.get_available_actions()
            print('Available actions:', available_actions)
            action = policy.forward(observation, available_actions)
            observation, reward, done, info = env.step(action)
            print(f'Taking action "{escape(action)}" -> Reward = {reward}')
            if done:
                break
    finally:
        env.close()