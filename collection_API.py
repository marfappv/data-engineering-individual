import requests
import time
import pandas as pd

def fetch_collections(page, limit, collections):
    url = "https://testnets-api.opensea.io/api/v1/collections?offset={}&limit={}".format(page*limit, limit)
    
    response = requests.request("GET", url)

    collections = response.json()['collections']

    for cs in response.json()['collections']:
        collections.append(transform(cs))

def transform(collection):

    out = {}
    out['collection_status'] = collection['safelist_request_status']
    out['only_proxied_transfers'] = collection['only_proxied_transfers']
    out['is_subject_to_whitelist'] = collection['is_subject_to_whitelist']
    out['opensea_buyer_fee_basis_points'] = collection['opensea_buyer_fee_basis_points']
    out['opensea_seller_fee_basis_points'] = collection['opensea_seller_fee_basis_points']
    out['featured'] = collection['featured']
    out['hidden'] = collection['hidden']
    out['require_email'] = collection['require_email']
    out['image_url'] = collection['image_url']
    out['large_image_url'] = collection['large_image_url']
    out['slug'] = collection['slug']
    out['telegram'] = collection['telegram_url']
    out['wiki'] = collection['wiki_url']
    out['twitter'] = collection['twitter_username']
    out['instagram'] = collection['instagram_username']
    out['nsfw'] = collection['is_nsfw']

    contracts = collection['primary_asset_contracts']
    for item in contracts:
        out['collection_name'] = item['name']
        out['asset_contract_type'] = item['asset_contract_type']
        out['created_date'] = item['created_date']
        out['nft_version'] = item['nft_version']
        out['num_of_tokens'] = item['total_supply']
        out['owner_number'] = item['owner']

    finances = collection['stats']
    out['day_avg_price'] = finances['one_day_average_price']
    out['month_avg_price'] = finances['thirty_day_average_price']
    out['week_avg_price'] = finances['seven_day_average_price']
    out['total_volume'] = finances['total_volume']
    out['total_sales'] = finances['total_sales']
    out['total_supply'] = finances['total_supply']
    out['average_price'] = finances['average_price']
    out['max_price'] = finances['market_cap']
    out['min_price'] = finances['floor_price']

    return out


def main():
    unique_collections = []
    for page in range(0,1):
        fetch_collections(page, 50, unique_collections)
        time.sleep(1)

    # Create a pandas dataframe out of the list.
    collections_API_df = pd.DataFrame(unique_collections)
    print(collections_API_df)

    # Save the dataframe in Parquet format.
    collections_API_df.to_parquet('parquet-files/collections_API.parquet', engine='fastparquet')

if __name__ == "__main__":
    main()