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

import sys
import json
from tqdm import tqdm
sys.path.insert(0, '../')

from web_agent_site.utils import DEFAULT_FILE_PATH, DEFAULT_ATTR_PATH
from web_agent_site.engine.engine import load_products

all_products, *_ = load_products(filepath=DEFAULT_FILE_PATH, attrpath=DEFAULT_ATTR_PATH)


docs = []
for p in tqdm(all_products, total=len(all_products)):
    option_texts = []
    options = p.get('options', {})
    for option_name, option_contents in options.items():
        option_contents_text = ', '.join(option_contents)
        option_texts.append(f'{option_name}: {option_contents_text}')
    option_text = ', and '.join(option_texts)

    doc = dict()
    doc['id'] = p['asin']
    doc['contents'] = ' '.join([
        p['Title'],
        p['Description'],
        p['BulletPoints'][0],
        option_text,
    ]).lower()
    doc['product'] = p
    docs.append(doc)


with open('./resources_100/documents.jsonl', 'w+') as f:
    for doc in docs[:100]:
        f.write(json.dumps(doc) + '\n')

with open('./resources/documents.jsonl', 'w+') as f:
    for doc in docs:
        f.write(json.dumps(doc) + '\n')

with open('./resources_1k/documents.jsonl', 'w+') as f:
    for doc in docs[:1000]:
        f.write(json.dumps(doc) + '\n')

with open('./resources_100k/documents.jsonl', 'w+') as f:
    for doc in docs[:100000]:
        f.write(json.dumps(doc) + '\n')
