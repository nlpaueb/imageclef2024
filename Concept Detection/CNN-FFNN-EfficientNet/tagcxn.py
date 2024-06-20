import gc
import json
import os

import numpy as np
from tqdm import tqdm

import utilities as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf


class TagCXN:  # base model class

    def __init__(self, configuration):
        self.configuration = configuration
        self.backbone = self.configuration['model']['backbone'] 
        self.preprocessor = self.configuration['model']['preprocessor']
        self.train_images_folder = self.configuration['data']['train_images_folder']
        self.val_images_folder = self.configuration['data']['val_images_folder']
        self.test_images_folder = self.configuration['data']['test_images_folder']
        self.train_data_path = self.configuration['data']['train_data_path']
        self.val_data_path = self.configuration['data']['val_data_path']
        self.test_data_path = self.configuration['data']['test_data_path']
        self.img_size = self.configuration['data']['img_size']
        self.train_data, self.val_data, self.test_data = dict(), dict(), dict()
        self.train_img_index, self.train_concepts_index = dict(), dict()
        self.val_img_index, self.val_concepts_index = dict(), dict()
        self.test_img_index = dict()
        self.tags_list = list()

        self.model = None

    def init_structures(self, skip_head=False, split_token='\t'):
        if '.csv' in self.train_data_path:
            self.train_data = self.load_csv_data(self.train_data_path, skip_head=skip_head, 
                                                 split_token=split_token)
        else:
            self.train_data = self.load_json_data(self.train_data_path)
        if '.csv' in self.val_data_path:
            self.val_data = self.load_csv_data(self.val_data_path, skip_head=skip_head,
                                               split_token=split_token)
        else:
            self.val_data = self.load_json_data(self.val_data_path)
        if '.csv' in self.test_data_path:
            self.test_data = self.load_csv_data(self.test_data_path, skip_head=skip_head, 
                                                split_token=split_token)
        else:
            self.test_data = self.load_json_data(self.test_data_path)

        print('Number of training examples:', len(self.train_data), 'Number of validation examples:',
              len(self.val_data), 'Number of testing examples:',
              len(self.test_data))

        self.train_img_index, self.train_concepts_index = utils.create_index(self.train_data)
        self.val_img_index, self.val_concepts_index = utils.create_index(self.val_data)
        self.test_img_index, _ = utils.create_index(self.test_data) 

        self.off_test_data = list()
        with open('../test_images.csv', 'r') as f:
            next(f)
            for line in f:
                self.off_test_data.append(str(line).split('\n', 1)[0])
        print('Number of test instances:', len(self.off_test_data))
        self.off_test_img_index = dict(zip(range(len(self.off_test_data)), list(self.off_test_data)))

        self.tags_list = self.load_tags(self.train_data)

        # remove modalities
        # modalities = ['X-Ray Computed Tomography', 'Computed Tomography', 'Ultrasonography', 'Magnetic Resonance Imaging', 'CT']
        # modalities = ['C0041618', 'C0024485', 'C0040405', 'C1699633']
        # self.tags_list = list(
        #     set(self.tags_list) - set(modalities)
        # )

        print('Number of categories:', len(self.tags_list))

    @staticmethod
    def load_csv_data(file_name, skip_head=False, split_token='\t'):
        """
        loads .csv file into a Python dictionary.
        :param file_name: the path to the file (string)
        :param skip_head: whether to skip the first row of the file (if there is a header) (boolean)
        :return: data dictionary (dict)
        """
        data = dict()
        with open(file_name, 'r') as f:
            if skip_head:
                next(f)
            for line in f:
                image = line.replace('\n', '').split(split_token)
                concepts = image[1].split(';')
                if image[0]:
                    data[str(image[0] + '.jpg')] = ';'.join(concepts)
        print('Data loaded from:', file_name)
        return data

    @staticmethod
    def load_json_data(file_name):
        """
        loads the data of JSON format into a Python dictionary
        :param file_name: the path to the file (string)
        :return: data dictionary (dict)
        """
        print('Data loaded from:', file_name)
        og = json.load(open(file=file_name, mode='r'))
        data = dict()
        for img in og:
            if 'normal' in og[img] and len(og[img]) == 1:
                og[img].remove('normal')
            data[img] = og[img]
        return data

    @staticmethod
    def load_tags(training_data):
        """
        loads the tags list
        :param training_data: training dictionary
        :return: the tags list
        """
        # if not isinstance(tags, list):
        #     return [line.strip() for line in open(tags, 'r')]
        tags = list()
        for img in training_data:
            if isinstance(training_data[img], str):
                tags.extend(training_data[img].split(';'))
            else:
                tags.extend(training_data[img])
        tags = set(tags)
        return list(tags)

    def build_model(self, pooling, repr_dropout=0., mlp_hidden_layers=None,
                    mlp_dropout=0., use_sam=False, batch_size=None, data_format='channels_last'):
        """
        builds the Keras model
        :param pooling: global pooling method (string)
        :param repr_dropout: whether to apply dropout to the encoder's representation (rate != 0) (float)
        :param mlp_hidden_layers: a list containing the
        number of units of the MLP head. Leave None for plain linear (list)
        :param mlp_dropout: whether to apply dropout to the MLPs layers (rate != 0) (float)
        :param use_sam: whether to use SAM optimization (boolean)
        :param batch_size: the batch size of training (int)
        :param data_format: whether the channels will be last
        :return: Keras model
        """
        if data_format == 'channels_first':
            inp = tf.keras.layers.Input(shape=self.img_size[::-1], name='input')
            x = self.backbone(inp, training=False).last_hidden_state
        else:
            inp = tf.keras.layers.Input(shape=self.img_size, name='input')
            x = self.backbone(self.backbone.input, training=False)

        encoder = tf.keras.Model(inputs=self.backbone.input, outputs=x, name='backbone')
        z = encoder(inp)
        if pooling == 'avg':
            z = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool', data_format=data_format)(z)
        elif pooling == 'max':
            z = tf.keras.layers.GlobalMaxPooling2D(name='max_pool', data_format=data_format)(z)
        else:
            z = utils.GeM(name='gem_pool', data_format=data_format)(z)

        if repr_dropout != 0.:
            z = tf.keras.layers.Dropout(rate=repr_dropout, name='repr_dropout')(z)
        for i, units in enumerate(mlp_hidden_layers):
            z = tf.keras.layers.Dense(units=units, activation='relu', name=f'MLP-layer-{i}')(z)
            if mlp_dropout != 0.:
                z = tf.keras.layers.Dropout(rate=mlp_dropout, name=f'MLP-dropout-{i}')(z)

        z = tf.keras.layers.Dense(units=len(self.tags_list), activation='sigmoid', name='LR')(z)
        model = tf.keras.Model(inputs=inp, outputs=z, name='TagCXN')
        print(model.summary())
        if use_sam:
            assert batch_size // 4 == 0  # this must be divided exactly due to tf.split in the implementation of SAM.
            model = tf.keras.models.experimental.SharpnessAwareMinimization(
                model=model, num_batch_splits=(batch_size // 4), name='TagCXN_w_SAM'
            )
        return model

    def train(self, train_parameters):
        """
        method that trains the model
        :param train_parameters: model and training hyperparameters
        :return: a Keras history object
        """
        batch_size = train_parameters.get('batch_size')
        self.model = self.build_model(pooling=train_parameters.get('pooling'),
                                      repr_dropout=train_parameters.get('repr_dropout'),
                                      mlp_hidden_layers=train_parameters.get('mlp_hidden_layers'),
                                      mlp_dropout=train_parameters.get('mlp_dropout'),
                                      use_sam=train_parameters.get('use_sam'), batch_size=batch_size,
                                      data_format=self.configuration['model']['data_format'])
        # loss = None
        if train_parameters.get('loss', {}).get('name') == 'bce':
            loss = tf.keras.losses.BinaryCrossentropy()
        elif train_parameters.get('loss', {}).get('name') == 'focal':
            loss = tf.keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=True, alpha=train_parameters.get('loss', {}).get('focal_alpha'),
                gamma=train_parameters.get('loss', {}).get('focal_gamma')
            )
        elif train_parameters.get('loss', {}).get('name') == 'asl':
            loss = utils.AsymmetricLoss(
                gamma_neg=train_parameters.get('loss', {}).get('asl_gamma_neg'),
                gamma_pos=train_parameters.get('loss', {}).get('asl_gamma_pos'),
                clip=train_parameters.get('loss', {}).get('asl_clip')
            )
        else:
            loss = utils.loss_1mf1_by_bce

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=train_parameters.get('learning_rate')),
            loss=loss
        )

        early_stopping = utils.ReturnBestEarlyStopping(monitor='val_loss',
                                                       mode='min',
                                                       patience=train_parameters.get('patience_early_stopping'),
                                                       restore_best_weights=True, verbose=1)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1,
                                                         patience=train_parameters.get('patience_reduce_lr'))

        print('\nTraining...')
        history = self.model.fit(
            self.train_generator(list(self.train_img_index), batch_size, self.tags_list),
            steps_per_epoch=np.ceil(len(self.train_img_index) / batch_size),
            validation_data=self.val_generator(list(self.val_img_index), batch_size, self.tags_list),
            validation_steps=np.ceil(len(self.val_img_index) / batch_size),
            callbacks=[early_stopping, reduce_lr], verbose=1, epochs=train_parameters['epochs']
        )
        print('\nEnd of training...')

        if train_parameters.get('checkpoint_path') is not None:
            self.model.save(train_parameters.get('checkpoint_path'))

        gc.collect()

        return history

    def train_generator(self, ids, batch_size, train_tags):
        """
        generator for training data
        :param ids: indices for each training sample in a batch (list)
        :param batch_size: batch size (int)
        :param train_tags: list of tags
        :return: yields a batch of data
        """
        batch = list()
        while True:
            np.random.shuffle(ids)
            for i in ids:
                batch.append(i)
                if i != len(ids):  # if not in the end of the list
                    if len(batch) == batch_size:
                        yield utils.load_batch(batch, self.train_img_index, self.train_concepts_index,
                                               self.train_images_folder, train_tags,
                                               self.preprocessor, size=self.img_size)

                        batch *= 0
                else:
                    yield utils.load_batch(batch, self.train_img_index, self.train_concepts_index,
                                           self.train_images_folder, train_tags, self.preprocessor, size=self.img_size)
                    batch *= 0

    def val_generator(self, ids, batch_size, train_tags):
        """
        generator for validation data
        :param ids: indices for each validation sample in a batch (list)
        :param batch_size: batch size (int)
        :param train_tags: list of tags
        :return: yields a batch of data
        """
        batch = list()
        while True:
            np.random.shuffle(ids)
            for i in ids:
                batch.append(i)
                if i != len(ids):
                    if len(batch) == batch_size:
                        yield utils.load_batch(batch, self.val_img_index,
                                               self.val_concepts_index, self.val_images_folder,
                                               train_tags, self.preprocessor, size=self.img_size, )
                        batch *= 0
                else:
                    yield utils.load_batch(batch, self.val_img_index, self.val_concepts_index,
                                           self.val_images_folder, train_tags, self.preprocessor, size=self.img_size, )
                    batch *= 0

    def test_generator(self, ids, index, batch_size, t='val'):
        """
        generator for testing data
        :param ids: indices for each testing sample in a batch (list)
        :param index: data index (dict)
        :param batch_size: batch size (int)
        :param t: flag for validation or testing (string)
        :return:
        """
        batch = list()
        while True:
            # np.random.shuffle(ids)
            for i in ids:
                batch.append(i)
                if i != len(ids):
                    if len(batch) == batch_size:
                        yield utils.load_test_batch(batch, index, self.val_images_folder
                                                    if t == 'val' else self.test_images_folder,
                                                    self.preprocessor, size=self.img_size)
                        batch *= 0
                else:
                    yield utils.load_test_batch(batch, index, self.val_images_folder
                                                if t == 'val' else self.test_images_folder,
                                                self.preprocessor, size=self.img_size)
                    batch *= 0

    def train_tune_test(self):
        """
        core logic of the file --> train, tune and test
        :return: a test score float, the test results in a dictionary format and a textual summary
        """
        self.init_structures(skip_head=self.configuration['data']['skip_head'], 
                             split_token=self.configuration['data']['split_token'])

        train_parameters = self.configuration['training_parameters']
        train_parameters.update(self.configuration['model_parameters'])

        training_history = self.train(train_parameters=train_parameters)

        bs = list(utils.divisor_generator(len(self.val_img_index)))[1]
        val_predictions = self.model.predict(self.test_generator(list(self.val_img_index),
                                                                 self.val_img_index, bs),
                                             verbose=1,
                                             steps=np.ceil(len(self.val_img_index) / bs))
        print(val_predictions.shape)
        best_threshold, val_score = self.tune_threshold(predictions=val_predictions,
                                                        not_bce=False)
        # best_threshold, val_score = self.tune_thresholds_hybrid(
        #     predictions=val_predictions,
        # )

        # y_pred_train = dict()
        # bs = list(utils.divisor_generator(len(self.train_img_index)))[1]
        # train_predictions = self.model.predict(self.test_generator(list(self.train_img_index),
        #                                                          self.train_img_index, bs),
        #                                        verbose=1,
        #                                        steps=np.ceil(len(self.train_img_index) / bs))
        # print(train_predictions.shape)
        # for i in tqdm(range(len(train_predictions))):
        #     predicted_tags = list()
        #     # bt = best_threshold
        #     for j in range(len(self.tags_list)):
        #         if train_predictions[i, j] >= best_threshold:
        #             predicted_tags.append(str(self.tags_list[j]))
        #     temp = ';'.join(predicted_tags)
        #     y_pred_train[str(list(self.train_data)[i]).split('.', 1)[0]] = temp
        # print('\n\nSaving results...\n')
        # with open('cnn_ffnn_densenet_train_pred.csv', 'w') as out_test:
        #     for result in y_pred_train:
        #         out_test.write(result + ',' + y_pred_train[result] + '\n')
        # print('Results saved!')
        # del train_predictions, y_pred_train
        y_pred_val = dict()

        # for i in tqdm(range(len(val_predictions))):
        #     predicted_tags = list()
        #     # bt = best_threshold
        #     for j in range(len(self.tags_list)):
        #         if val_predictions[i, j] >= best_threshold:
        #             predicted_tags.append(str(self.tags_list[j]))
        #     temp = ';'.join(predicted_tags)
        #     y_pred_val[str(list(self.val_data)[i]).split('.', 1)[0]] = temp
        # print('\n\nSaving results...\n')
        # with open('cnn_ffnn_densenet_val_pred.csv', 'w') as out_test:
        #     for result in y_pred_val:
        #         out_test.write(result + ',' + y_pred_val[result] + '\n')
        # print('Results saved!')
        del val_predictions, y_pred_val


        bs = list(utils.divisor_generator(len(self.test_img_index)))[1]
        test_predictions = self.model.predict(self.test_generator(list(self.test_img_index),
                                                                  self.test_img_index, bs, t='test'),
                                              verbose=1,
                                              steps=np.ceil(len(self.test_img_index) / bs))
        print(test_predictions.shape)
        test_score, test_results = self.test(best_threshold=best_threshold,
                                             predictions=test_predictions, )
        # test_score, test_results = self.test_w_knn(best_threshold=best_threshold, 
        #                                            predictions=test_predictions)
        # test_score, test_results = self.test_hybrid(t1=0.4, t2=0.2, 
        #                                             predictions=test_predictions)
        del test_predictions


        ###### official test predictions ######
        bs = list(utils.divisor_generator(len(self.off_test_img_index)))[1]
        off_test_predictions = self.model.predict(self.test_generator(list(self.off_test_img_index),
                                                                  self.off_test_img_index, bs, t='test'),
                                                  verbose=1,
                                                  steps=np.ceil(len(self.off_test_img_index) / bs))
        print(off_test_predictions.shape)
        y_pred_off_test = dict()
        for i in tqdm(range(len(off_test_predictions))):
            predicted_tags = list()
            # bt = best_threshold
            for j in range(len(self.tags_list)):
                if off_test_predictions[i, j] >= best_threshold:
                    predicted_tags.append(str(self.tags_list[j]))
            temp = ';'.join(predicted_tags)
            y_pred_off_test[str(list(self.off_test_data)[i]).split('.', 1)[0]] = temp
        print('\n\nSaving official test results...\n')
        with open('cnn_ffnn_densenet_off_test_pred_2.csv', 'w') as out_test:
            for result in y_pred_off_test:
                out_test.write(result + ',' + y_pred_off_test[result] + '\n')
        print('Results saved!')
        del off_test_predictions, y_pred_off_test
        ###### official test predictions ######

        s = ('Development score = ' + str(test_score) +
             ' with threshold = ' + str(best_threshold) + ' and validation score = ' + str(val_score))

        return test_score, test_results, best_threshold, s

    def run(self):
        """
        basic run method
        :return: a dictionary of checkpoint paths alongside with scores and thresholds
        """
        thresholds_map = dict()
        test_scores = list()
        info = list()

        test_score, test_results, best_threshold, txt = self.train_tune_test()
        test_scores.append(test_score)
        if self.configuration['training_parameters']['checkpoint_path'] is not None:
            thresholds_map[self.configuration['training_parameters']['checkpoint_path']] = [best_threshold, test_score]
        info.append(txt)
        for i in range(len(info)):
            print(info[i])
        # s = 'Mean dev score was: ' + str(sum(test_scores) / len(test_scores)) + '\n\n\n'
        # print(s)
        info *= 0
        test_scores *= 0
        #
        if self.configuration.get('save_results'):
            print('\n\nSaving results...\n')
            with open(self.configuration.get('results_path'), 'w') as out_test:
                for result in test_results:
                    out_test.write(str(result).split('.', 1)[0] + ',' + test_results[result] + '\n')
            print('Results saved!')

        # pickle.dump(thresholds_map, open(str(self.backbone_name) + '_map.pkl', 'wb'))
        # pickle.dump(thresholds_map, open('temp_map.pkl', 'wb'))
        return thresholds_map

    def tune_threshold(self, predictions, not_bce=False):
        """
        method that tunes the classification threshold
        :param predictions: array of validation predictions (NumPy array)
        :param not_bce: flag for not bce losses (boolean)
        :return: best threshold and best validation score
        """
        print('\nGot predictions for validation set.')
        # steps = 100
        init_thr = 0.1
        if not_bce:
            init_thr = 0.3
        f1_scores = dict()
        recalls = dict()
        precisions = dict()
        print('Initial threshold:', init_thr)
        for i in tqdm(np.arange(init_thr, 1., 0.01)):
            threshold = i
            # print('Current checking threshold =', threshold)
            y_pred_val = dict()
            for j in range(len(predictions)):
                predicted_tags = list()

                # indices of elements that are above the threshold.
                for index in np.argwhere(predictions[j] >= threshold).flatten():
                    predicted_tags.append(self.tags_list[index])

                # string with ';' after each tag. Will be split in the f1 calculations.
                # print(len(predicted_tags))
                y_pred_val[list(self.val_data.keys())[j]] = ';'.join(predicted_tags)
                # print(y_pred_val)

            f1_scores[threshold], p, r, _ = utils.evaluate_f1(self.val_data, y_pred_val, test=True)
            recalls[threshold] = r
            precisions[threshold] = p

        # get key with max value.
        best_threshold = max(f1_scores, key=f1_scores.get)
        print('The best F1 score on validation data' +
              ' is ' + str(f1_scores[best_threshold]) +
              ' achieved with threshold = ' + str(best_threshold) + '\n')

        # print('Recall:', recalls[best_threshold], ' Precision:', precisions[best_threshold])
        return best_threshold, f1_scores[best_threshold]

    def tune_thresholds_hybrid(self, predictions, ):
        import pickle, itertools
        from collections import Counter

        print('\nGot predictions for validation set.')
        print('\nTuning thresholds...')

        def get_embeddings(data):
            embeddings = dict()
            for img_name in tqdm(data):
                if 'train' in img_name:
                    images_path = '../train/'
                else:
                    images_path = '../valid/'
                path = os.path.join(images_path, img_name)
                img = tf.keras.preprocessing.image.load_img(path, target_size=self.img_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = self.preprocessor.preprocess_input(img_array)
                embedding = self.model.predict(img_array, verbose=0).T.flatten()
                embedding = embedding / np.linalg.norm(embedding)
                embeddings[img_name] = embedding
            return embeddings
        
        def knn_retrieve_tags(best_k, best_r, val_embedding, train_embeddings):
            train_embeddings_array = np.array([train_embeddings[i] for i in train_embeddings])
            ids = [i for i in train_embeddings]
            sims = train_embeddings_array @ val_embedding.flatten()
            sims = np.array(sims).flatten()

            top_k = np.argsort(sims)[-best_k:]
            tags_list = list()
            for i, index in enumerate(top_k):
                tags = self.train_data[ids[index]].split(';')
                for tag in tags:
                    tags_list.append(tag)
            most_frequent_tags = Counter(tags_list).most_common(best_r)

            return set([t[0] for t in most_frequent_tags])
        
        # train_embeddings = get_embeddings(self.train_data)
        train_embeddings = pickle.load(open('../embeddings_foivos/embedding_dict_train_ordered2024.pkl', 'rb'))
        new_embeddings_dict_train = dict()
        for image_name, embedding in train_embeddings.items():
            new_image_name = image_name + '.jpg'
            new_embeddings_dict_train[new_image_name] = embedding.T.flatten() / np.linalg.norm(embedding.T.flatten())
        train_embeddings = dict(new_embeddings_dict_train)

        # val_embeddings = get_embeddings(self.val_data)
        val_embeddings = pickle.load(open('../embeddings_foivos/embedding_dict_val_ordered2024.pkl', 'rb'))
        new_embeddings_dict_val = dict()
        for image_name, embedding in val_embeddings.items():
            new_image_name = image_name + '.jpg'
            new_embeddings_dict_val[new_image_name] = embedding.T.flatten() / np.linalg.norm(embedding.T.flatten())
        val_embeddings = dict(new_embeddings_dict_val)

        init_thr = (0.1, 0.)
        f1_scores = dict()
        recalls = dict()
        precisions = dict()
        print('Initial thresholds:', init_thr)

        threshold_combinations = list(itertools.combinations(np.arange(init_thr[0], 1., 0.1), 2))

        for t2, t1 in tqdm(threshold_combinations):
            y_pred_val = dict()

            for j in range(len(predictions)):
                embedding_j = val_embeddings[list(val_embeddings.keys())[j]] 
                # tags_gt_t1 = set()  # Tags greater than t1
                predicted_tags_gt_t1 = list()
                predicted_tags_grey_zone = list() # Tags in t2 - t1 zone grey zone we will call knn later

                indices_in_zone = np.argwhere((predictions[j] >= t2) & (predictions[j] < t1)).flatten()  # grey zone

                #tags greater than t1
                for index in np.argwhere(predictions[j] >= t1).flatten():
                    # tags_gt_t1.add(self.tags_list[index])
                    predicted_tags_gt_t1.append(self.tags_list[index])

                # Append tags within the [t2,t1] grey zone
                for index in indices_in_zone:
                    # tags_grey_zone.add(self.tags_list[index])
                    predicted_tags_grey_zone.append(self.tags_list[index])
                
                if predicted_tags_grey_zone:
                    knn_tags = knn_retrieve_tags(best_k=5, best_r=2, val_embedding=embedding_j, train_embeddings=train_embeddings)
                    common_tags = set(predicted_tags_grey_zone) & knn_tags
                    predicted_tags_gt_t1.extend(list(common_tags))
                    # for tag in common_tags:
                    #     predicted_tags_gt_t1.append(tag)
                
                y_pred_val[list(self.val_data.keys())[j]] = ';'.join(list(set(predicted_tags_gt_t1)))
            
            f1_scores[(t1, t2)], p, r, _ = utils.evaluate_f1(self.val_data, y_pred_val, test=True)
            recalls[(t1, t2)] = r
            precisions[(t1, t2)] = p
        
        best_thresholds = max(f1_scores, key=f1_scores.get)
        print('The best F1 score on validation data' +
              ' is ' + str(f1_scores[best_thresholds]) +
              ' achieved with thresholds = ' + str(best_thresholds) + '\n')
        
        return best_thresholds, f1_scores[best_thresholds]

    def test(self, best_threshold, predictions):
        """
        method that performs the evaluation on the test data
        :param best_threshold: the tuned classification threshold (float)
        :param predictions: 2D array of test predictions (NumPy array)
        :return: test score and test results dictionary
        """
        print('\nStarting evaluation on test set...')
        y_pred_test = dict()

        for i in tqdm(range(len(predictions))):
            predicted_tags = list()
            # bt = best_threshold
            for j in range(len(self.tags_list)):
                if predictions[i, j] >= best_threshold:
                    predicted_tags.append(str(self.tags_list[j]))

            # string! --> will be split in the f1 function

            # final_tags = list(set(set(predicted_tags).union(set(most_frequent_tags))))
            # temp = ';'.join(final_tags)
            temp = ';'.join(predicted_tags)
            y_pred_test[list(self.test_data)[i]] = temp

        f1_score, p, r, _ = utils.evaluate_f1(self.test_data, y_pred_test, test=True)
        print('\nThe F1 score on the test set is: {}\n'.format(f1_score))
        # print('Precision score:', p)
        # print('Recall score:\n', r)
        # pickle.dump(y_pred_test, open(f'my_test_results_split_{split}.pkl', 'wb'))
        return f1_score, y_pred_test
    
    def test_hybrid(self, t1, t2, predictions):
        import pickle
        from collections import Counter

        print('\nStarting evaluation on test set...')

        def get_embeddings(data):
            embeddings = dict()
            for img_name in tqdm(data):
                if 'train' in img_name:
                    images_path = '../train/'
                else:
                    images_path = '../valid/'
                path = os.path.join(images_path, img_name)
                img = tf.keras.preprocessing.image.load_img(path, target_size=self.img_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = self.preprocessor.preprocess_input(img_array)
                embedding = self.model.predict(img_array, verbose=0).T.flatten()
                embedding = embedding / np.linalg.norm(embedding)
                embeddings[img_name] = embedding
            return embeddings
        
        def knn_retrieve_tags(best_k, best_r, test_embedding, train_embeddings):
            train_embeddings_array = np.array([train_embeddings[i] for i in train_embeddings])
            ids = [i for i in train_embeddings]
            sims = train_embeddings_array @ test_embedding.flatten()
            sims = np.array(sims).flatten()

            top_k = np.argsort(sims)[-best_k:]
            tags_list = list()
            for i, index in enumerate(top_k):
                tags = self.train_data[ids[index]].split(';')
                for tag in tags:
                    tags_list.append(tag)
            most_frequent_tags = Counter(tags_list).most_common(best_r)

            return set([t[0] for t in most_frequent_tags])
        
        # train_embeddings = get_embeddings(self.train_data)
        train_embeddings = pickle.load(open('../embeddings_foivos/embedding_dict_train_ordered2024.pkl', 'rb'))
        new_embeddings_dict_train = dict()
        for image_name, embedding in train_embeddings.items():
            new_image_name = image_name + '.jpg'
            new_embeddings_dict_train[new_image_name] = embedding.T.flatten() / np.linalg.norm(embedding.T.flatten())
        train_embeddings = dict(new_embeddings_dict_train)

        # test_embeddings = get_embeddings(self.test_data)
        test_embeddings = pickle.load(open('../embeddings_foivos/embedding_dict_dev_ordered2024.pkl', 'rb'))
        new_embeddings_dict_test = dict()
        for image_name, embedding in test_embeddings.items():
            new_image_name = image_name + '.jpg'
            new_embeddings_dict_test[new_image_name] = embedding.T.flatten() / np.linalg.norm(embedding.T.flatten())
        test_embeddings = dict(new_embeddings_dict_test)

        y_pred_test = dict()
        avg_clf_tags, avg_knn_match_tags = list(), list()
        for i in tqdm(range(len(predictions))):
            embedding_i = test_embeddings[list(test_embeddings.keys())[i]] 
            
            predicted_tags = list()
            
            predicted_tags_grey_zone = list()

            indices_in_zone = np.argwhere((predictions[i] >= t2) & (predictions[i] < t1)).flatten()  # grey zone
            for index in indices_in_zone:
                # tags_grey_zone.add(self.tags_list[index])
                predicted_tags_grey_zone.append(self.tags_list[index])

            for j in range(len(self.tags_list)):
                if predictions[i, j] >= t1:  # confident
                    predicted_tags.append(str(self.tags_list[j]))

            avg_clf_tags.append(len(predicted_tags))        
            
            if predicted_tags_grey_zone:
                knn_tags = knn_retrieve_tags(best_k=76, best_r=2, test_embedding=embedding_i, 
                                             train_embeddings=train_embeddings)
                common_tags = set(predicted_tags_grey_zone) & knn_tags  # intersection of knn and grey tags
                avg_knn_match_tags.append(len(list(common_tags)))
                predicted_tags.extend(list(common_tags))
        
            y_pred_test[list(self.test_data.keys())[i]] = ';'.join(list(set(predicted_tags)))

        f1_score, p, r, _ = utils.evaluate_f1(self.test_data, y_pred_test, test=True)
        print('\nThe F1 score on the test set is: {}\n'.format(f1_score))
        print('Avg clf tags:', sum(avg_clf_tags) / len(avg_clf_tags), ' and avg knn matched tags:', sum(avg_knn_match_tags) / len(avg_knn_match_tags))
        pickle.dump(avg_clf_tags, open('avg_clf_tags.pkl', "wb"))
        pickle.dump(avg_knn_match_tags, open('avg_knn_match_tags.pkl', "wb"))
        return f1_score, y_pred_test
    
    def test_w_knn(self, best_threshold, predictions):
        import pickle
        from collections import Counter

        print('\nStarting evaluation on test set...')

        def get_embeddings(data):
            embeddings = dict()
            for img_name in tqdm(data):
                if 'train' in img_name:
                    images_path = '../train/'
                else:
                    images_path = '../valid/'
                path = os.path.join(images_path, img_name)
                img = tf.keras.preprocessing.image.load_img(path, target_size=self.img_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = self.preprocessor.preprocess_input(img_array)
                embedding = self.model.predict(img_array, verbose=0).T.flatten()
                embedding = embedding / np.linalg.norm(embedding)
                embeddings[img_name] = embedding
            return embeddings
        
        def knn_retrieve_tags(best_k, best_r, test_embedding, train_embeddings):
            train_embeddings_array = np.array([train_embeddings[i] for i in train_embeddings])
            ids = [i for i in train_embeddings]
            sims = train_embeddings_array @ test_embedding.flatten()
            sims = np.array(sims).flatten()

            top_k = np.argsort(sims)[-best_k:]
            tags_list = list()
            for i, index in enumerate(top_k):
                tags = self.train_data[ids[index]].split(';')
                for tag in tags:
                    tags_list.append(tag)
            most_frequent_tags = Counter(tags_list).most_common(best_r)

            return set([t[0] for t in most_frequent_tags])

        def knn_retrieve_tags1or2(best_k, test_embedding, train_embeddings):
            # modalities = ['C0040405', 'C1699633', 'C0041618', 'C0024485']

            train_embeddings_array = np.array([train_embeddings[i] for i in train_embeddings])
            ids = [i for i in train_embeddings]
            sims = train_embeddings_array @ test_embedding.flatten()
            sims = np.array(sims).flatten()

            top_k = np.argsort(sims)[-best_k:]
            tags_list = list()
            for i, index in enumerate(top_k):
                tags = self.train_data[ids[index]].split(';')
                for tag in tags:
                    tags_list.append(tag)
            
            tag_counts = Counter(tags_list)
            top_tag, top_count = tag_counts.most_common(1)[0]

            second_tag, second_count = tag_counts.most_common(2)[-1]
            threshold = 0.58  
            if (top_count - second_count) / top_count < threshold:
                top_tags = [top_tag, second_tag]
            else:
                top_tags = [top_tag]
            
            return set(top_tags)

        def knn_retrieve_tags_more(best_k, test_embedding, train_embeddings):
            train_embeddings_array = np.array([train_embeddings[i] for i in train_embeddings])
            ids = [i for i in train_embeddings]
            sims = train_embeddings_array @ test_embedding.flatten()
            sims = np.array(sims).flatten()

            top_k = np.argsort(sims)[-best_k:]
            tags_list = list()
            for i, index in enumerate(top_k):
                tags = self.train_data[ids[index]].split(';')
                for tag in tags:
                    tags_list.append(tag)
            
            tag_counts = Counter(tags_list)
            top_tag, top_count = tag_counts.most_common(1)[0]

            second_tag, second_count = tag_counts.most_common(2)[-1]
            threshold = 0.58  
            if (top_count - second_count) / top_count < threshold:
                top_tags = [top_tag, second_tag]
            else:
                top_tags = [top_tag]
            
            threshold2 = 0.65
            # If the second and third tags are very close, include the third tag as well
            third_tag, third_count = tag_counts.most_common(3)[-1]
            if (top_count - third_count) / top_count < threshold2:
                top_tags.append(third_tag)

            # If the third and fourth tags are very close, include the fourth tag as well
            fourth_tag, fourth_count = tag_counts.most_common(4)[-1]
            if (top_count - fourth_count) / top_count < threshold2:
                top_tags.append(fourth_tag)

            return set(top_tags)

        
        # train_embeddings = get_embeddings(self.train_data)
        train_embeddings = pickle.load(open('../embeddings_foivos/embedding_dict_train_ordered2024.pkl', 'rb'))
        new_embeddings_dict_train = dict()
        for image_name, embedding in train_embeddings.items():
            new_image_name = image_name + '.jpg'
            new_embeddings_dict_train[new_image_name] = embedding.T.flatten() / np.linalg.norm(embedding.T.flatten())
        train_embeddings = dict(new_embeddings_dict_train)

        # test_embeddings = get_embeddings(self.test_data)
        test_embeddings = pickle.load(open('../embeddings_foivos/embedding_dict_dev_ordered2024.pkl', 'rb'))
        new_embeddings_dict_test = dict()
        for image_name, embedding in test_embeddings.items():
            new_image_name = image_name + '.jpg'
            new_embeddings_dict_test[new_image_name] = embedding.T.flatten() / np.linalg.norm(embedding.T.flatten())
        test_embeddings = dict(new_embeddings_dict_test)

        y_pred_test = dict()
        for i in tqdm(range(len(predictions))):
            embedding_i = test_embeddings[list(test_embeddings.keys())[i]] 
            
            predicted_tags = list()
            

            for j in range(len(self.tags_list)):
                if predictions[i, j] >= best_threshold:  # confident
                    predicted_tags.append(str(self.tags_list[j]))   
            
            if not predicted_tags:
                # knn_tags = knn_retrieve_tags(best_k=33, best_r=1, test_embedding=embedding_i, 
                #                              train_embeddings=train_embeddings)
                # knn_tags = knn_retrieve_tags1or2(best_k=33, test_embedding=embedding_i,
                #                                  train_embeddings=train_embeddings)
                knn_tags = knn_retrieve_tags_more(best_k=33, test_embedding=embedding_i,
                                                  train_embeddings=train_embeddings)
                predicted_tags.extend(list(knn_tags))
        
            y_pred_test[list(self.test_data.keys())[i]] = ';'.join(list(set(predicted_tags)))

        f1_score, p, r, _ = utils.evaluate_f1(self.test_data, y_pred_test, test=True)
        return f1_score, y_pred_test
                

                    




