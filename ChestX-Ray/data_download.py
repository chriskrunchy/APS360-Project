import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Define the dataset
dataset = 'nih-chest-xrays/data'

# Download the dataset
api.dataset_download_files(dataset, path='nih-chest-xray-dataset', unzip=True)
