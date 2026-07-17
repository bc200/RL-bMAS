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

__version__ = '0.4.2'

import os
from os.path import join as pjoin

from alfworld.utils import mkdirs


_default_alfworld_cache = os.path.expanduser("~/.cache/alfworld")
ALFWORLD_DATA = mkdirs(os.getenv("ALFWORLD_DATA", _default_alfworld_cache))
os.environ["ALFWORLD_DATA"] = ALFWORLD_DATA  # Set the environment variable, in case it wasn't.

BUILTIN_DATA_PATH = pjoin(os.path.dirname(__file__), "data")
ALFRED_PDDL_PATH = pjoin(BUILTIN_DATA_PATH, 'alfred.pddl')
ALFRED_TWL2_PATH = pjoin(BUILTIN_DATA_PATH, 'alfred.twl2')
