from helium import *
from selenium import webdriver
import pandas as pd

#chrome_options = webdriver.ChromeOptions()
#chrome_options.add_argument('--headless')
#chrome_options.add_argument('--no-sandbox')
#chrome_options.add_argument('--disable-dev-shm-usage')
#driver = webdriver.Chrome('chromedriver', options=chrome_options)

#set_driver(driver)
driver = start_chrome()

def get_details(url):
    go_to(url)
    creator_name,name,collection_name,price,fav = None,None,None,None,None
    try:
      creator_name = Text(to_right_of="Created by").web_element.text
    except Exception:
      pass

    try:
      name = S('//h1[contains(@class,"title")]').web_element.text
    except Exception:
      pass
    
    try:
      collection_name = S('//div[contains(@class,"item--collection-info")]').web_element.text.split('\n')[0]
    except Exception:
      pass

    try:
      price = S('//div[contains(@class,"Price--amount")][contains(@tabindex,"-1")]').web_element.text
    except Exception:
      pass

    try:
      fav = S('//button[contains(@aria-label,"Favorited by")]').web_element.text.split("\n")[1].split(" ")[0]
    except Exception:
      pass
    
    details = {
      'creator' : creator_name,
      'artwork_name' : name,
      'collection_name' : collection_name,
      'price' : price,
      'likes' : fav,
      'currency': 'ETH'
    }
    return details

go_to("https://opensea.io/assets?search[sortAscending]=false&search[sortBy]=LISTING_DATE")

import time

final_links = set()

for _ in range(5):
    links = find_all(S("//a[contains(@href,'assets/')]"))
    for i in list(map(lambda x:x.web_element.get_attribute("href"),links)):
        final_links.add(i)
    scroll_down(1000)
    time.sleep(1)

data = list(map(get_details,final_links))

 # Create a pandas dataframe out of the list.
opensea_ws_df = pd.DataFrame(data)

# Save the dataframe in Parquet format.
opensea_ws_df.to_parquet('parquet-files/opensea_ws.parquet', engine='fastparquet')