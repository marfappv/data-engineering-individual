import requests

url = "https://testnets-api.opensea.io/api/v1/collections?offset=0&limit=1"

response = requests.request('GET', url)

collections = response.json()['collections']

for c in response.json()['collections']:
    collections.append(c)

contracts = c['primary_asset_contracts']
print(contracts)