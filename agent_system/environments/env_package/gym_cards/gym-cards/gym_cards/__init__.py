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

from gym_cards.envs.points import Point24Env
from gym_cards.envs.ezpoints import EZPointEnv
from gym_cards.envs.blackjack import BlackjackEnv
from gym_cards.envs.numberline import NumberLineEnv
from gymnasium.envs.registration import register

register(
    id='gym_cards/Blackjack-v0',
    entry_point='gym_cards.envs:BlackjackEnv',
    max_episode_steps=300,
)

register(
    id='gym_cards/Points24-v0',
    entry_point='gym_cards.envs:Point24Env',
    max_episode_steps=300,
)

register(
    id='gym_cards/EZPoints-v0',
    entry_point='gym_cards.envs:EZPointEnv',
    max_episode_steps=300,
)

register(
    id='gym_cards/NumberLine-v0',
    entry_point='gym_cards.envs:NumberLineEnv',
    max_episode_steps=300,
)