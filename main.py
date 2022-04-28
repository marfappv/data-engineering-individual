# Part 1: Setup

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd
from pandas import Series, DataFrame
from pandas.api.types import CategoricalDtype
pd.options.display.max_columns = None
import numpy as np; np.random.seed(2022)
import boto3
import json
import sagemaker.amazon.common as smac
import sagemaker
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter
from sagemaker.analytics import HyperparameterTuningJobAnalytics, TrainingJobAnalytics
from sagemaker.amazon.amazon_estimator import get_image_uri
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import Image as image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from numpy import mean
from numpy import std
import psycopg2 as pg
from nbformat import current
import itertools as it
import io
import os
import sys
import time
from scipy.sparse import lil_matrix

sns.set(color_codes=True)
sns.set_context('paper')
five_thirty_eight = ["#30a2da", "#fc4f30", "#e5ae38", "#6d904f", "#8b8b8b",]
sns.set_palette(five_thirty_eight)
%matplotlib inline

bucket = 'sagemaker-us-west-2-369454669781'
prefix = 'sagemaker/fm-recsys'
role = 'arn:aws:iam::369454669781:role/service-role/AmazonSageMaker-ExecutionRole-20220415T151752'
sess = sagemaker.Session()
smclient = boto3.Session().client('sagemaker')

# Part 2: Data Preparation
engine = pg.connect("dbname='opensea' user='marfapopova21' host='opensea.c5pkb2dzarva.us-west-2.rds.amazonaws.com' port='5432' password='qwerty123'")
df1 = pd.read_sql('select * from nfts.collections', con=engine)
df1 = df1.dropna(how='all', axis=1)
df1 = df1.dropna(how='all')

engine = pg.connect("dbname='opensea' user='marfapopova21' host='opensea.c5pkb2dzarva.us-west-2.rds.amazonaws.com' port='5432' password='qwerty123'")
df2 = pd.read_sql('select * from nfts.finances', con=engine)
df2 = df2.dropna(how='all', axis=1)
df2 = df2.dropna(how='all')

df = pd.merge(df1, df2, on = 'collection_name')
df = df.drop(['owner_number', 'max_price', 'min_price'], axis=1)
df = df.drop_duplicates()
df.drop(columns = ['created_date'], inplace = True)
df['collection_status'].replace(['not_requested', 'requested', 'approved','verified'], [0,1,2,3], inplace=True)
df['asset_contract_type'].replace(['non-fungible', 'semi-fungible'], [0,1], inplace=True)
df = df[df['nft_version'].notna()]
df = df[df['tokens'].notna()]
df['nft_version'] = df['nft_version'].astype(float)
df['tokens'] = df['tokens'].astype(int)
df['opensea_buyer_fee_basis_points'] = df['opensea_buyer_fee_basis_points'].astype(float)
df['opensea_seller_fee_basis_points'] = df['opensea_seller_fee_basis_points'].astype(float)

from sklearn.preprocessing import RobustScaler
stratified_df = df.drop(columns = ['collection_name'])
stratified_df = pd.DataFrame(RobustScaler().fit_transform(stratified_df), columns=stratified_df.columns)

train_df, validate_df, test_df = np.split(stratified_df .sample(frac=1), [int(.6*len(stratified_df )), int(.8*len(stratified_df ))])

all_df = pd.concat([train_df, validate_df, test_df])
# Choose a response variable
Y = all_df['total_sales']

# Drop a response variable and collection name from the feature data set
X = all_df.drop(columns = ['total_sales'])

nb_total_sales = np.unique(Y.values).shape[0]
nb_independent_vars = np.unique(X.values).shape[0]
feature_dim = nb_total_sales + nb_independent_vars

def convert_sparse_matrix(df, nb_rows, nb_total_sales, nb_independent_vars):
    # dataframe to array
    df_val = df.values

    # determine feature size
    nb_cols = nb_total_sales + nb_independent_vars
    print("# of rows = {}".format(str(nb_rows)))
    print("# of cols = {}".format(str(nb_cols)))

    # extract customers and ratings
    df_X = df_val[:, 0:2]
    # Features are one-hot encoded in a sparse matrix
    X = lil_matrix((nb_rows, nb_cols)).astype('float32')
    df_X[:, 1] = nb_total_sales + df_X[:, 1]
    coords = df_X[:, 0:2]
    X[np.arange(nb_rows), coords[:, 0]] = 1
    X[np.arange(nb_rows), coords[:, 1]] = 1

    # create label with ratings
    Y = df_val[:, 2].astype('float32')

    # validate size and shape
    print(X.shape)
    print(Y.shape)
    assert X.shape == (nb_rows, nb_cols)
    assert Y.shape == (nb_rows, )
    return X, Y

print("Convert training data set to one-hot encoded sparse matrix")
train_X, train_Y = convert_sparse_matrix(train_df, train_df.shape[0], nb_total_sales, nb_independent_vars)
print("Convert validation data set to one-hot encoded sparse matrix")
validate_X, validate_Y = convert_sparse_matrix(validate_df, validate_df.shape[0], nb_total_sales, nb_independent_vars)
print("Convert test data set to one-hot encoded sparse matrix")
test_X, test_Y = convert_sparse_matrix(test_df, test_df.shape[0], nb_total_sales, nb_independent_vars)

def save_as_protobuf(X, Y, bucket, key):
    buf = io.BytesIO()
    smac.write_spmatrix_to_sparse_tensor(buf, X, Y)
    buf.seek(0)
    obj = '{}'.format(key)
    boto3.resource('s3', region_name='us-west-2').Bucket(bucket).Object(obj).upload_fileobj(buf)
    return 's3://{}/{}'.format(bucket, obj)

s3_train_path = save_as_protobuf(train_X, train_Y, bucket, 'prepare/train/train.protobuf')
print("Training data set in protobuf format uploaded at {}".format(s3_train_path))
s3_val_path = save_as_protobuf(validate_X, validate_Y, bucket, 'prepare/validate/validate.protobuf')
print("Validation data set in protobuf format uploaded at {}".format(s3_val_path))

def chunk(x, batch_size):
    chunk_range = range(0, x.shape[0], batch_size)
    chunks = [x[p: p + batch_size] for p in chunk_range]
    return chunks

test_x_chunks = chunk(test_X, 10000)
test_y_chunks = chunk(test_Y, 10000)
N = len(test_x_chunks)
for i in range(N):
    test_data = save_as_protobuf(
        test_x_chunks[i],
        test_y_chunks[i],
        bucket,
        "prepare/test/test_" + str(i) + ".protobuf")
    print(test_data)

# Part 3: Model Training
container = get_image_uri(boto3.Session().region_name, 'factorization-machines')

%time

output_location = 's3://{}/train/'.format(bucket)
s3_train_path = 's3://{}/prepare/train/train.protobuf'.format(bucket)
s3_val_path = 's3://{}/prepare/validate/validate.protobuf'.format(bucket)

fm = sagemaker.estimator.Estimator(container,
                                   role, 
                                   train_instance_count=1, 
                                   train_instance_type='ml.c5.4xlarge',
                                   output_path=output_location,
                                   sagemaker_session=sess)

fm.set_hyperparameters(feature_dim=228,
                      predictor_type='regressor',
                      mini_batch_size=200,
                      num_factors=64,
                      bias_lr=0.02,
                      epochs=10)

fm.fit({'train': s3_train_path,'test': s3_val_path}, wait=False)

training_job_name = fm._current_job_name
metric_name = 'train:rmse:epoch'

# run this cell to check current status of training job
fm_training_job_result = smclient.describe_training_job(TrainingJobName=training_job_name)

status = fm_training_job_result['TrainingJobStatus']
if status != 'Completed':
    print('Reminder: the training job has not been completed.')
else:
    print('The training job is completed.')

rmse_dataframe = TrainingJobAnalytics(training_job_name=training_job_name,metric_names=[metric_name]).dataframe()
rmse_dataframe

