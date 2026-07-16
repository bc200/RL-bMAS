# import pandas as pd
# import glob

# # 读取所有CSV文件
# files = glob.glob('*.csv')
# dfs = [pd.read_csv(f)[['question', 'answer']] for f in files if 'question' in pd.read_csv(f).columns]
# print(len(dfs[0]))
# # 合并并保存
# pd.concat(dfs, ignore_index=True).to_csv('merge.csv', index=False)

import pandas as pd
import glob
import numpy as np

# 读取并提取数据
files = glob.glob('*.csv')
# dfs = [pd.read_csv(f)[['question', 'answer']] 
#        for f in files if all(col in pd.read_csv(f).columns 
#        for col in ['question', 'answer'])]
#从challenge_test.csv中提取600条，从其他文件中提取前1200条
dfs = []
for f in files:
    df = pd.read_csv(f)
    if all(col in df.columns for col in ['question', 'answer']):
        if f == 'challenge_test.csv':
            #将question和options合并成一个question列
            df['question'] = df['question'] + "\nOptions\n" + df['options']
            dfs.append(df[['question', 'answer']].head(400))
        elif f == 'test.csv':
            dfs.append(df[['question', 'answer']].head(1000))
        elif f == 'hle_choice.csv':
            dfs.append(df[['question', 'answer']].head(200))
        else:
            dfs.append(df[['question', 'answer']].head(1200))

# 合并、打乱、保存
df = pd.concat(dfs, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)



df.to_csv('merge.csv', index=False)