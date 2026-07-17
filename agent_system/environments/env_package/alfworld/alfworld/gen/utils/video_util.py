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

import subprocess
import alfworld.gen.constants as constants

class VideoSaver(object):

    def __init__(self, frame_rate=constants.VIDEO_FRAME_RATE):
        self.frame_rate = frame_rate

    def save(self, image_path, save_path):
        subprocess.call(["ffmpeg -r %d -pattern_type glob -y -i '%s' -c:v libx264 -pix_fmt yuv420p '%s'" %
                         (self.frame_rate, image_path, save_path)], shell=True)