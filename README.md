# Project description and instruction for launching

## Description
This project is an extension of data engineering group project done previously, as part of accrediatation for Data Engineering course at UCL. Whilst the coursework scope stays the same, this project:
* Successfully merges 3 parquet files scraped from 2 websites: [Open Sea](https://opensea.io), [NFT Showroom](https://nftshowroom.com), and adds 4 more data tables into the RDS.
* Deploys machine learning pipeline, predicting a number of total sales of an NFT collection, based on its features. The sales vary from 0 to 10.

The project's objective is to create more NFT datasets currently lacking on [Kaggle](https://www.kaggle.com/search?q=nft+in%3Adatasets) and eradicate the problem of weak labelling in the digital art industry.


## Instruction
This is a guide on how to run the project using your Docker.
1. Type the following in your terminal:
```git clone https://github.com/marfappv/data_eng_ind.git```

2. Make sure ```dockerfile``` is run properly from ```python-docker``` folder. It will install all necessary libraries to run the Machine Learning pipeline code.

3. Type the following in your terminal:
```python3 main.py```