import requests

url = "https://testnets-api.opensea.io/api/v1/collections?offset=0&limit=2"

response = requests.request('GET', url)

collections = response.json()['collections']

for contract in collections:
    c = contract['primary_asset_contracts'] 

for items in c:
    print('collection_name', items['name'])
    print('asset_contract_type', items['asset_contract_type'])
    print('created_date', items['created_date'])
    print('nft_version', items['nft_version'])
    print('owner_number', items['owner'])
    print('tokens', items['total_supply'])

for finances in collections:
    s = finances['stats']

print('day_avg_price', s['one_day_average_price'])
print('week_avg_price', s['seven_day_average_price'])
print('month_avg_price', s['thirty_day_average_price'])
print('total_volume', s['total_volume'])
print('total_sales', s['total_sales'])
print('total_supply', s['total_supply'])
print('average_price', s['average_price'])
print('max_price', s['market_cap'])
print('min_price', s['floor_price'])

