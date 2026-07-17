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
Model implementations. The model interface should be suitable for both
the ``site env'' and the ``text env''.
"""
import random

random.seed(4)


class BasePolicy:
    def __init__(self):
        pass
    
    def forward(observation, available_actions):
        """
        Args:
            observation (`str`):
                HTML string

            available_actions ():
                ...
        Returns:
            action (`str`): 
                Return string of the format ``action_name[action_arg]''.
                Examples:
                    - search[white shoes]
                    - click[button=Reviews]
                    - click[button=Buy Now]
        """
        raise NotImplementedError


class HumanPolicy(BasePolicy):
    def __init__(self):
        super().__init__()

    def forward(self, observation, available_actions):
        action = input('> ')
        return action


class RandomPolicy(BasePolicy):
    def __init__(self):
        super().__init__()
    
    def forward(self, observation, available_actions):
        if available_actions['has_search_bar']:
            action = 'search[shoes]'
        else:
            try:
                action_arg = random.choice(available_actions['clickables'])
            except:
                action_arg = 'None'
            action = f'click[{action_arg}]'
        return action
