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

from alfworld.gen.agents.semantic_map_planner_agent import SemanticMapPlannerAgent


class DeterministicPlannerAgent(SemanticMapPlannerAgent):
    def __init__(self, thread_id=0, game_state=None):
        super(DeterministicPlannerAgent, self).__init__(thread_id, game_state)
        self.action_sequence = None
        self.question = None

    def reset(self, seed=None, info=None, scene=None, objs=None):
        info = super(DeterministicPlannerAgent, self).reset(seed, info, scene=scene, objs=objs)
        self.action_sequence = ['Plan', 'End']
        return info

    def step(self, action, executing_plan=True):
        if not executing_plan:
            self.action_sequence = self.action_sequence[1:]
        super(DeterministicPlannerAgent, self).step(action)

    def get_action(self, action_ind=None):
        assert(action_ind is None)
        return {'action': self.action_sequence[0]}

    def get_reward(self):
        return 0, self.terminal

