import os
os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/temurin-11.jdk/Contents/Home"
os.environ["SPARK_HOME"] = "/Users/Marfa-Popova/data_eng_ind/spark-3.2.1-bin-hadoop3.2"
os.environ["AWS_ACCESS_KEY_ID"] = "AKIAVMBJW37K3DWZCMHQ"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import findspark
findspark.init("/Users/Marfa-Popova/data_eng_ind/spark-3.2.1-bin-hadoop3.2")
import pyspark
from pyspark.sql import SQLContext

sc = pyspark.SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

assets = sqlContext.read.parquet("s3a://data-eng-ind/parquet-files/opensea_API.parquet", header=True)
collections = sqlContext.read.parquet("s3a://dataenggroup/parquet-files/opensea_API.parquet", header=True)
finances = sqlContext.read.parquet("s3a://dataenggroup/parquet-files/opensea_API.parquet", header=True)
socials = sqlContext.read.parquet("s3a://dataenggroup/parquet-files/opensea_API.parquet", header=True)
urls = sqlContext.read.parquet("s3a://dataenggroup/parquet-files/opensea_API.parquet", header=True)

# Connecting to Postgres
#psql --host=opensea.ckmusmy93z05.eu-west-2.rds.amazonaws.com --port=5432 --username=marfapopova21 --password --dbname=opensea

postgres_uri = "jdbc:postgresql://opensea.ckmusmy93z05.eu-west-2.rds.amazonaws.com:5432/opensea"
user = "marfapopova21"
password = "qwerty123"

assets.write \
    .format("jdbc") \
    .mode("append") \
    .option("url", postgres_uri) \
    .option("dbtable", "nft.assets") \
    .option("user", user) \
    .option("password", password) \
    .option("driver", "org.postgresql.Driver") \
    .save()

collections.write \
    .format("jdbc") \
    .mode("append") \
    .option("url", postgres_uri) \
    .option("dbtable", "nft.collections") \
    .option("user", user) \
    .option("password", password) \
    .option("driver", "org.postgresql.Driver") \
    .save()

finances.write \
    .format("jdbc") \
    .mode("append") \
    .option("url", postgres_uri) \
    .option("dbtable", "nft.finances") \
    .option("user", user) \
    .option("password", password) \
    .option("driver", "org.postgresql.Driver") \
    .save()

socials.write \
    .format("jdbc") \
    .mode("append") \
    .option("url", postgres_uri) \
    .option("dbtable", "nft.socials") \
    .option("user", user) \
    .option("password", password) \
    .option("driver", "org.postgresql.Driver") \
    .save()

urls.write \
    .format("jdbc") \
    .mode("append") \
    .option("url", postgres_uri) \
    .option("dbtable", "nft.urls") \
    .option("user", user) \
    .option("password", password) \
    .option("driver", "org.postgresql.Driver") \
    .save()