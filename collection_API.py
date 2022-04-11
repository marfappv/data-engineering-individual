import requests
#import time
#import pandas as pd

#def fetch_collecions(page, limit, collections):
 #   url = "https://testnets-api.opensea.io/api/v1/collections?offset={}&limit={}".format(page*limit, limit)
    
  #  response = requests.request("GET", url)
    
   # for c in response.json()["collections"]:
    #    collections.append(transform(c))

url = "https://testnets-api.opensea.io/api/v1/collections?offset=0&limit=1"

response = requests.request("GET", url)

for c in response.json()["collections"]:
    out = {}
    out['created_date'] = c['primary_asset_contracts']['created_date']
    out['name'] = c['primary_asset_contracts']['created_date']['name']
    out['nft_version'] = c['primary_asset_contracts']['created_date']['nft_version']
#    out['owner'] = c['primary_asset_contracts']['name']
#    out['featured'] = c['featured']
#    out['num_owners'] = c['stats']['num_owners']
#    out['nsfw'] = c['is_nsfw']
    print(out)

#def transform(collection):
#    out = {}

#    out['created_date'] = c['primary_asset_contracts']['created_date']
    
  #  creator = collection['creator']
   # if creator is not None and creator['user'] is not None and creator['user']['username'] is not None:
    #  out['creator'] = creator['user']['username']
    #else:
      #out['creator'] = 'unknown'
    
#    out['name'] = c['primary_asset_contracts']['name']
 #   out['owner'] = c['primary_asset_contracts']['name']
#   out['num_owners'] = c['stats']['num_owners']
 #   out['featured'] = c['featured']
#    out['nsfw'] = c['is_nsfw']
#
    #return out
 #   
#def main():
#    nfts = []
#   for page in range(0,20):
#        fetch_assets(page, 200, nfts)
#        time.sleep(1)
#
#    # Create a pandas dataframe out of the list.
 #   opensea_API_df = pd.DataFrame(nfts)
  #  print(opensea_API_df)

    # Save the dataframe in Parquet format.
   # opensea_API_df.to_parquet('parquet-data/opensea_API.parquet', engine='fastparquet')
#
#if __name__ == "__main__":
 #   main()