"""
@author: Carmel WENGA
This python file define useful function to download and load the googlenet model into a tensorflow session.
For more details about these functions, please refer to the tutorial's notebook of this repository.
"""

import tensorflow as tf
import tarfile
import urllib
import sys
import os

import settings

class GoogLeNet:

    def __init__(self):

        # url to the googlenet model
        self.model_url = settings.ENV_GOOGLENET_MODEL_URL

        # folder where to save the tar file to download
        self.save_dir = settings.ENV_GOOGLENET_DIR

        # name of the downloaded file
        self.name = settings.ENV_GOOGLENET_ARCHIVE_MODEL_FILE

        # location of the model
        self.model = settings.ENV_GOOGLENET_MODEL_GRAPH


    def download_googlenet(self):
        """
        Download and extract googlenet model from the downloaded tar file 
        """
        
        # create the save dir if it not already exists
        os.makedirs(self.save_dir, exist_ok=True)   
        
        # file path
        fpath = os.path.join(self.save_dir, self.name)
        
        # define a progress function
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r[INFO] Downloading %s %.1f%%' % (self.name, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        
        # download with urllib
        fpath, _ = urllib.request.urlretrieve(self.model_url, fpath, _progress)
        
        print()
        statinfo = os.stat(fpath)
        print('Successfully downloaded', self.name, statinfo.st_size, 'bytes.')
        with tarfile.open(fpath, 'r:gz') as tar:
            tar.extractall(self.save_dir, filter='data')

    def load_model(self):
        """
        Load the model from the .pb file of GoogLeNet.
        """
        
        # download the model if not already downloaded
        if not os.path.exists(self.model):
            GoogLeNet.download_googlenet(self)
        
        print('[INFO] Loading the model ...')
        
        # Creates graph from saved graph_def.pb.
        with tf.io.gfile.GFile(self.model, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')