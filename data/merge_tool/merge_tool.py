import pandas as pd
import glob
import os

train_dfs = []
test_dfs = []

def split_train_test(df, train_ratio=0.2):
    """
    按顺序切分：
    前20%作为训练集，剩余作为测试集
    """
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    return train_df, test_df


# ========== 处理 CSV 文件 ==========
csv_files = glob.glob('*.csv')

for f in csv_files:
    try:
        df = pd.read_csv(f)

        if all(col in df.columns for col in ['question', 'answer']):
            # 按文件名决定取多少条
            if os.path.basename(f) == 'hle-choice.csv':
                temp = df[['question', 'answer']].head(500).copy()
                temp['answer_type'] = 'multiple'
                temp['decision_prompt'] = "This is a Single choice question question. There is one and only one correct answer.\nOutput your answers in the format:\n\\{{the final answer is boxed[answer]}} at the end of your response.\nFor instance, if there were 7 options and the answers were C, the output would be:\n\\{{the final answer is boxed[C]}}\n"
                train_df, test_df = split_train_test(temp, train_ratio=0.2)

            elif os.path.basename(f) == 'MMLU_Pro_subset.csv':
                temp = df[['question', 'answer']].head(600).copy()
                temp['answer_type'] = 'multiple'
                temp['decision_prompt'] = "This is a Single choice question question. There is one and only one correct answer.\nOutput your answers in the format:\n\\{{the final answer is boxed[answer]}} at the end of your response.\nFor instance, if there were 7 options and the answers were C, the output would be:\n\\{{the final answer is boxed[C]}}\n"
                train_df, test_df = split_train_test(temp, train_ratio=1)

            elif os.path.basename(f) == 'test.csv':
                temp = df[['question', 'answer']].head(500).copy()
                temp['answer_type'] = 'multiple'
                temp['decision_prompt'] = "This is a Single choice question question. There is one and only one correct answer.\nOutput your answers in the format:\n\\{{the final answer is boxed[answer]}} at the end of your response.\nFor instance, if there were 7 options and the answers were C, the output would be:\n\\{{the final answer is boxed[C]}}\n"
                train_df, test_df = split_train_test(temp, train_ratio=0.2)

            else:
                temp = df[['question', 'answer']].head(1200).copy()
                temp['answer_type'] = 'multiple'
                temp['decision_prompt'] = "This is a Single choice question question. There is one and only one correct answer.\nOutput your answers in the format:\n\\{{the final answer is boxed[answer]}} at the end of your response.\nFor instance, if there were 7 options and the answers were C, the output would be:\n\\{{the final answer is boxed[C]}}\n"
                train_df, test_df = split_train_test(temp, train_ratio=0.2)

            

            # 切分训练集和测试集
            

            train_dfs.append(train_df)
            test_dfs.append(test_df)

        else:
            print(f"CSV 文件缺少 question 或 answer 列: {f}")

    except Exception as e:
        print(f"读取 CSV 文件失败: {f}, 错误: {e}")


# ========== 处理 JSONL 文件 ==========
jsonl_files = glob.glob('*.jsonl')

for f in jsonl_files:
    try:
        df = pd.read_json(f, lines=True)

        if all(col in df.columns for col in ['question', 'answer']):
            # 按文件名决定取多少条
            if os.path.basename(f) == 'frames_test.jsonl':
                temp = df[['question', 'answer']].head(800).copy()
                temp['answer_type'] = 'single'
                temp['decision_prompt'] = "Output your answers in the format:\n\\{{the final answer is boxed[answer]}} at the end of your response.\n"
                train_df, test_df = split_train_test(temp, train_ratio=0.25)
            elif os.path.basename(f) == 'browse_comp_test.jsonl':
                temp = df[['question', 'answer']].head(700).copy()
                temp['answer_type'] = 'single'
                temp['decision_prompt'] = "Output your answers in the format:\n\\{{the final answer is boxed[answer]}} at the end of your response.\n"
                train_df, test_df = split_train_test(temp, train_ratio=1)
            elif os.path.basename(f) == 'gaia_text.jsonl':
                temp = df[['question', 'answer']].head(100).copy()
                temp['answer_type'] = 'single'
                temp['decision_prompt'] = "Output your answers in the format:\n\\{{the final answer is boxed[answer]}} at the end of your response.\n"
                train_df, test_df = split_train_test(temp, train_ratio=1)
            elif os.path.basename(f) == 'hotpotqa_500.jsonl':
                temp = df[['question', 'answer']].head(500).copy()
                temp['answer_type'] = 'single'
                temp['decision_prompt'] = "Output your answers in the format:\n\\{{the final answer is boxed[answer]}} at the end of your response.\n"
                train_df, test_df = split_train_test(temp, train_ratio=1)
            elif os.path.basename(f) == 'musique_500.jsonl':
                temp = df[['question', 'answer']].head(500).copy()
                temp['answer_type'] = 'single'
                temp['decision_prompt'] = "Output your answers in the format:\n\\{{the final answer is boxed[answer]}} at the end of your response.\n"
                train_df, test_df = split_train_test(temp, train_ratio=0.2)
            else:
                temp = df[['question', 'answer']].head(1200).copy()
                temp['answer_type'] = 'single'
                temp['decision_prompt'] = "Output your answers in the format:\n\\{{the final answer is boxed[answer]}} at the end of your response.\n"
                train_df, test_df = split_train_test(temp, train_ratio=0.2)
                
            
            # 切分训练集和测试集
            

            train_dfs.append(train_df)
            test_dfs.append(test_df)

        else:
            print(f"JSONL 文件缺少 question 或 answer 列: {f}")

    except Exception as e:
        print(f"读取 JSONL 文件失败: {f}, 错误: {e}")


# ========== 分别合并训练集和测试集 ==========
if train_dfs or test_dfs:
    train_merged = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame(columns=['question', 'answer', 'answer_type'])
    test_merged = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame(columns=['question', 'answer', 'answer_type'])

    # 清理空值
    train_merged = train_merged.dropna(subset=['question', 'answer'])
    test_merged = test_merged.dropna(subset=['question', 'answer'])

    # 转字符串
    for col in ['question', 'answer', 'answer_type']:
        train_merged[col] = train_merged[col].astype(str)
        test_merged[col] = test_merged[col].astype(str)

    # 分别打乱
    train_merged = train_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    test_merged = test_merged.sample(frac=1, random_state=42).reset_index(drop=True)

    # 训练集在前，测试集在后
    final_merged = pd.concat([train_merged, test_merged], ignore_index=True)

    # 保存
    final_merged.to_csv('merge_tool.csv', index=False, encoding='utf-8-sig')

    print(f"训练集条数: {len(train_merged)}")
    print(f"测试集条数: {len(test_merged)}")
    print(f"总条数: {len(final_merged)}")
    print("已保存到 merge.csv")

else:
    print("没有可合并的数据文件")
