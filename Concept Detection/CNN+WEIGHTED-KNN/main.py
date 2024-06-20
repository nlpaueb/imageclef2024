from tagcxn import TagCXN
from tagcxn_multiclass import TagCXNModalities
from knn import KNN
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf






knn_configuration = {

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

    'encoder': {
        'backbone': 'densenet_2024.hdf5',
        'preprocessor': tf.keras.applications.densenet,  # accompanying preprocessor
    },

    'knn_params': {
        'k_range': (0, 100),
        'k_step': 10,
        'r_range': (1, 5),
        'weighted': True
    },

    'embeddings': {
        'load_embeddings': True,
        'train_path': '../embeddings_foivos/embedding_dict_train_ordered2024.pkl',
        'val_path': '../embeddings_foivos/embedding_dict_val_ordered2024.pkl',
        'test_path': '../embeddings_foivos/embedding_dict_dev_ordered2024.pkl',
    },
    'save_results': True,
    'results_path': 'knn_learnable_weights_dev_pred_1.csv'  # path to the results file

}

k = KNN(
    configuration=knn_configuration
)
k.run()