from tagcxn import TagCXN
from knn import KNN
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

from transformers import AutoImageProcessor, TFConvNextModel

configuration = {
    'data': {
        'train_images_folder': '../train/',  # path to folder of training images
        'val_images_folder': '../train/',  # path to folder of validation images
        'test_images_folder': '../train/',  # path to folder of testing images
        'train_data_path': '../stratified_data/flamel_csv/train_mysplit_processed.csv',  # .csv file or json file
        'val_data_path': '../stratified_data/flamel_csv/valid_mysplit_processed.csv',  # .csv file or json file
        'test_data_path': '../stratified_data/flamel_csv/dev_mysplit_processed.csv',  # .csv file or json file
        'skip_head': True,

	    'split_token': ',',
        'img_size': (224, 224, 3),
    },
    'model': {
        # any instance of tf.keras.Model with its output being the representation of the image (not logits)
        # if a HuggingFace model is selected, data_format must be 'channels_first'
        'backbone': tf.keras.applications.DenseNet121(weights='imagenet', include_top=False,),
        'preprocessor': tf.keras.applications.densenet,  # accompanying preprocessor
        'data_format': 'channels_last'
    },
    'model_parameters': {
        'pooling': 'gem',
        'repr_dropout': 0.,
        'mlp_hidden_layers': [],  # no. of hidden layers
        'mlp_dropout': 0.,
        'use_sam': False,
    },
    'training_parameters': {
        'loss': {
            'name': 'bce'
        },
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'patience_early_stopping': 3,
        'patience_reduce_lr': 1,
        'checkpoint_path': None  # path to save the best model
    },
    'save_results': True,
    'results_path': 'cnn_ffnn_densenet_dev_pred_2.csv'  # path to the results file

}

t = TagCXN(
    configuration=configuration
)

t.run()
