import pandas as pd
from sympy import Q
df = pd.read_csv('data_sourcing/showroom/kaggle.csv')

# Drop categorical columns that are unnecessary
df = df.drop(columns = ['art_series', 'path', 'year', 'rights', 'cid', 'royalty', 'type'])

# Rename columns that may be interchangeably confusing
df = df.rename(columns={'title':'collection_name', 'name':'artwork_name', 'symbol':'currency'})

# Changing order for the columns
df = df.reindex(['creator', 'artwork_name', 'collection_name', 'price', 'currency', 'likes', 'tokens', 'nsfw'], axis=1)

#Drop index column
df.reset_index(drop=True, inplace=True)

#Convert the dataframe to a parquet file with no index column
df.to_parquet('parquet-files/showroom.parquet', index=False)