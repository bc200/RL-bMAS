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

def get_environment(env_type):
    if env_type == 'AlfredTWEnv':
        from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        return AlfredTWEnv
    elif env_type == 'AlfredThorEnv':
        from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
        return AlfredThorEnv
    elif env_type == 'AlfredHybrid':
        from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_hybrid import AlfredHybrid
        return AlfredHybrid
    else:
        raise NotImplementedError(f"Environment {env_type} is not implemented.")
