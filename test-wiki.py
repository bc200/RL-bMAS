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

import json
import os

import requests

PROXY = os.getenv("WIKIMEDIA_PROXY_URL")

headers = {
    "User-Agent": os.getenv("WIKIMEDIA_USER_AGENT", "verl-agent/1.0")
    # Wikipedia 要求格式: 工具名/版本 (联系方式)
}

try:
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action":"query","titles":"La Silvia opera composer","format":"json","prop":"extracts"},
        proxies={"http": PROXY, "https": PROXY} if PROXY else None,
        headers=headers,
        timeout=10
    )
    print(f"状态码: {r.status_code}")
    print(f"响应前200字符: {r.text[:200]}")
    data = r.json()
    print("✅ JSON解析成功！")
except Exception as e:
    print(f"❌ 错误: {e}")
