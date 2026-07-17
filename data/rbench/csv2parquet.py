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

from pathlib import Path
import sys
import pandas as pd

#!/usr/bin/env python3
# GitHub Copilot
# 将同目录的 rbench-t.csv 切分并保存为 train.parquet / test.parquet

# 确定脚本所在目录（若在交互环境中运行则使用当前工作目录）
script_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
csv_path = script_dir / "rbench-t.csv"
if not csv_path.exists():
    print(f"ERROR: 找不到文件 {csv_path}", file=sys.stderr)
    sys.exit(1)

# 读取 CSV
df = pd.read_csv(csv_path)

# 切分：前 200 条为 train，200-600（索引 200 到 599）为 test
train = df.iloc[:200].copy()
test = df.iloc[200:600].copy()

# 保存为 parquet（需要安装 pyarrow 或 fastparquet）
train_path = script_dir / "train.parquet"
test_path = script_dir / "test.parquet"

train.to_parquet(train_path, index=False)
test.to_parquet(test_path, index=False)

print(f"Saved {len(train)} rows -> {train_path}")
print(f"Saved {len(test)} rows -> {test_path}")