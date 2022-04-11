import requests

url = "https://testnets-api.opensea.io/api/v1/collections?offset=0&limit=1"

response = requests.request('GET', url)

collections = response.json()['collections']

for c in response.json()['collections']:
    collections.append(c)

for contract in collections:
    p_a_s = contract['primary_asset_contracts'] 

for items in p_a_s:
    print('collection_name', items['name'])
    print('asset_contract_type', items['asset_contract_type'])
    print('created_date', items['created_date'])
    print('nft_version', items['nft_version'])
    print('tokens', items['total_supply'])
    print('owner_number', items['owner'])

for finances in collections:
    f = finances['stats']

print('day_avg_price', f['one_day_average_price'])
print('week_avg_price', f['seven_day_average_price'])
print('month_avg_price', f['thirty_day_average_price'])
print('total_volume', f['total_volume'])
print('total_sales', f['total_sales'])
print('total_supply', f['total_supply'])
print('average_price', f['average_price'])
print('max_price', f['market_cap'])
print('min_price', f['floor_price'])

print('name', c['name'])
print('collection_status', c['safelist_request_status'])
print('only_proxied_transfers', c['only_proxied_transfers'])
print('is_subject_to_whitelist', c['is_subject_to_whitelist'])
print('opensea_buyer_fee_basis_points', c['opensea_buyer_fee_basis_points'])
print('opensea_seller_fee_basis_points', c['opensea_seller_fee_basis_points'])
print('featured', c['featured'])
print('hidden', c['hidden'])
print('require_email', c['require_email'])
print('image_url', c['image_url'])
print('large_image_url', c['large_image_url'])
print('slug', c['slug'])
print('telegram', c['telegram_url'])
print('wiki', c['wiki_url'])
print('twitter', c['twitter_username'])
print('instagram', c['instagram_username'])
print('nsfw', c['is_nsfw'])