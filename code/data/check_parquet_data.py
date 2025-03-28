import pandas as pd


datasets = ['bird', 'spider']
splits = ['train', 'test']


# for dataset in datasets:
#     for split in splits:
#         path = f'data/sql/{dataset}/{split}.parquet'
#         df = pd.read_parquet(path)
#         print(f"dataset: {dataset}, split: {split}, Loaded {len(df)} rows")
#         if split == 'test' and dataset == 'bird':
#             print(df.head(1)['prompt'][0][0]['content'])
#             break


path = f'data/sql/wor/bird_wor/train.parquet'
df = pd.read_parquet(path)
print(df.head(1)['prompt'][0][0]['content'])

# for i in range(len(df)):
#     # print(df.iloc[i]['prompt'])
#     print(df.iloc[i]['extra_info']['db_path'])
#     print('--------------------------------')
#     break
