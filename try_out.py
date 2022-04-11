import requests

url = "https://testnets-api.opensea.io/api/v1/collections?offset=0&limit=2"

response = requests.request('GET', url)

# Decode the JSON data into a dictionary
collections = response.json()['collections']

for contract in collections:
    x = contract['primary_asset_contracts'] 

for items in x:
    print('collection_name', items['name'])
    print('asset_contract_type', items['asset_contract_type'])
    print('created_date', items['created_date'])
    print('nft_version', items['nft_version'])
    print('owner_number', items['owner'])
    print('tokens', items['total_supply'])