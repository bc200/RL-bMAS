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

import random

from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv


class AlfredHybrid(object):
    '''
    Hybrid training manager for switching between AlfredTWEnv and AlfredThorEnv
    '''

    def __init__(self, config, train_eval="train"):
        print("Setting up AlfredHybrid env")
        self.hybrid_start_eps = config["env"]["hybrid"]["start_eps"]
        self.hybrid_thor_prob = config["env"]["hybrid"]["thor_prob"]

        self.config = config
        self.train_eval = train_eval

        self.curr_env = "tw"
        self.eval_mode = config["env"]["hybrid"]["eval_mode"]
        self.num_resets = 0

    def choose_env(self):
        if self.curr_env == "thor":
            return self.thor
        else:
            return self.tw

    def init_env(self, batch_size):
        alfred_tw_env = AlfredTWEnv(self.config, train_eval=self.train_eval)
        alfred_thor_env = AlfredThorEnv(self.config, train_eval=self.train_eval)

        self.batch_size = batch_size
        self.tw = alfred_tw_env.init_env(batch_size)
        self.thor = alfred_thor_env.init_env(batch_size)
        return self

    def seed(self, num):
        env = self.choose_env()
        return env.seed(num)

    def step(self, actions):
        env = self.choose_env()
        return env.step(actions)

    def reset(self):
        if "eval" in self.train_eval:
            assert(self.eval_mode in ['tw', 'thor'])
            self.curr_env = self.eval_mode
        else:
            if self.num_resets >= self.hybrid_start_eps:
                self.curr_env = "thor" if random.random() < self.hybrid_thor_prob else "tw"
            else:
                self.curr_env = "tw"
        env = self.choose_env()
        obs, infos = env.reset()
        self.num_resets += self.batch_size
        return obs, infos
