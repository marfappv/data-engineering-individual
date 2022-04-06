# Project description and instruction for launching

## Description
This project is an extension of data engineering group project done previously, as part of accrediatation for Data Engineering course at UCL. Whilst the coursework scope stays the same, this project:
* Successfully merges 3 parquet files scraped from 3 websites: [Open Sea](https://opensea.io), [NFT Showroom](https://nftshowroom.com), [Art Bloocks](https://www.artblocks.io).
* Deploys machine learning pipeline, enabling the algorithm to predict a price class of an NFT based on its feastures. The 3 price classes are cheap, average, and luxury.

The project's objective is to create more NFT datasets currently lacking on [Kaggle](https://www.kaggle.com/search?q=nft+in%3Adatasets) and eradicate the problem of weak labelling in the digital art industry.


## Instruction
This is a guide on how to run the project using local server. Copy-paste following lines **line by line** to your terminal, depending on your operating system. If the third step runs with an error, try moving on with the rest of commands, as you might have all the needed packages installed.

#### MacOS
```
git clone https://github.com/marfappv/data_eng_group
cd data_eng_group/environments
bash MacOs_env.sh
cd ..
python3 full_MacOS_script.py
```

#### Linux
```
git clone https://github.com/marfappv/data_eng_group
cd data_eng_group/environments
bash Linux_env.sh
cd ..
python3 full_MacOS_script.py
```

#### Check the imputed RDS

1. Type the following line to your teminal:
```
psql --host=nfts.cuweglfckgza.eu-west-2.rds.amazonaws.com --port=5432 --username=marfapopova21 --password --dbname=nfts
```
2. Insert password: qwerty123. Hit enter.
3. Type the following 3 lines in terminal:
```
\dn
\dt nfts.*
select * from nfts.assets;
```