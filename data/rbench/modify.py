from pathlib import Path
import sys
import pandas as pd

#!/usr/bin/env python3
# modify.py
# 读取同目录下的 test.parquet，将 question 与选项 A-F 合并为新的 question 文本，转存为 CSV


HERE = Path(__file__).resolve().parent
PARQUET_FILE = HERE / "test-00000-of-00001.parquet"
CSV_FILE = HERE / "rbench-t.csv"
OPTION_KEYS = ["A", "B", "C", "D", "E", "F"]

if not PARQUET_FILE.exists():
    print(f"ERROR: not found: {PARQUET_FILE}", file=sys.stderr)
    sys.exit(1)

# 读取 parquet
df = pd.read_parquet(PARQUET_FILE)

# 确保有 question 列
if "question" not in df.columns:
    print("ERROR: 'question' column not found in parquet file", file=sys.stderr)
    sys.exit(1)

# 合并 question 与选项
def merge_question(row):
    base = "" if pd.isna(row.get("question")) else str(row.get("question")).strip()
    parts = [base] if base else []
    parts.append("Options:")
    for key in OPTION_KEYS:
        val = row.get(key)
        if pd.notna(val):
            s = str(val).strip()
            if s:
                parts.append(f"{key}. {s}")
    return "\n".join(parts)

df["task"] = df.apply(merge_question, axis=1)
#新增一个键值decision_prompt
decision_prompt = "Output your answers in the format:\n\\{{the final answer is boxed[answer]}} at the end of your response.\nFor instance, if there were 7 options and the answers were C, the output would be:\n\\{{the final answer is boxed[C]}}"
df["decision_prompt"] = decision_prompt
df["answer_type"] = "multiple"

# 输出为 CSV，保留新的 task 和原有 answer 列（如果存在）
cols_to_save = ["task", "decision_prompt", "answer_type"]
if "answer" in df.columns:
    cols_to_save.append("answer")

df.to_csv(CSV_FILE, columns=cols_to_save, index=False, encoding="utf-8")
print(f"Saved merged CSV to: {CSV_FILE}")