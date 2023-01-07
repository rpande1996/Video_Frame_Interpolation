import wget
from zipfile import ZipFile
import os

ds_path = 'http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip'
dataset_dir = '../dataset/'
wget.download(ds_path, dataset_dir)

ds_file = 'vimeo_triplet.zip'
with ZipFile(dataset_dir+ds_file, 'r') as obj:
    obj.extractall(path=dataset_dir)

os.remove(dataset_dir+ds_file)