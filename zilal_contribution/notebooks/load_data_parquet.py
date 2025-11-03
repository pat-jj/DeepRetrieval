import pandas as pd
df = pd.read_parquet("/projects/illinois/eng/cs/jimeng/zelalae2/scratch/DeepRetrieval/code/data/search_engine/pubmed_32/train.parquet")
first4 = df.head(1)
print(first4.columns) # Column names
print(first4['input']) # Column names

