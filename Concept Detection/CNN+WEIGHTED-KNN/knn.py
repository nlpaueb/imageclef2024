import os

import numpy as np
from tqdm import tqdm
import json
from collections import Counter
import pickle
import utilities as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf


class KNN:

    def __init__(self, configuration):
        self.configuration = configuration
        self.backbone = self.configuration['encoder']['backbone'] 
        self.preprocessor = self.configuration['encoder']['preprocessor']
        self.train_images_folder = self.configuration['data']['train_images_folder']
        self.val_images_folder = self.configuration['data']['val_images_folder']
        self.test_images_folder = self.configuration['data']['test_images_folder']
        self.train_data_path = self.configuration['data']['train_data_path']
        self.val_data_path = self.configuration['data']['val_data_path']
        self.test_data_path = self.configuration['data']['test_data_path']
        self.img_size = self.configuration['data']['img_size']

        # knn tuning
        self.k_range = self.configuration['knn_params']['k_range']
        self.k_step = self.configuration['knn_params']['k_step']
        self.r_range = self.configuration['knn_params']['r_range']
        self.weighted = self.configuration['knn_params']['weighted']

        self.load_embeddings = self.configuration['embeddings']['load_embeddings']

        self.train_data, self.val_data, self.test_data = dict(), dict(), dict()
        # self.train_img_index, self.train_concepts_index = dict(), dict()
        # self.val_img_index, self.val_concepts_index = dict(), dict()
        # self.test_img_index = dict()
        self.tags_list = list()
        self.training_embeddings, self.validation_embeddings, self.test_embeddings = dict(), dict(), dict()

        self.encoder = None
        self.ids = None
        

    def init_structures(self, skip_head=False, split_token='\t'):
        """
        initializes data structures
        :param skip_head: skip input file header
        :param split_token: split token in the .csv input file
        :return:
        """
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

        # self.train_img_index, self.train_concepts_index = utils.create_index(self.train_data)
        # self.val_img_index, self.val_concepts_index = utils.create_index(self.val_data)
        # self.test_img_index, _ = utils.create_index(self.test_data) 

        self.off_test_data = list()
        with open('../test_images.csv', 'r') as f:
            next(f)
            for line in f:
                self.off_test_data.append(str(line).split('\n', 1)[0])
        print('Number of test instances:', len(self.off_test_data))  # has .jpg
        # print(self.off_test_data)
        # self.off_test_img_index = dict(zip(range(len(self.off_test_data)), list(self.off_test_data)))
        

        self.tags_list = self.load_tags(self.train_data)

        # remove modalities
        # modalities = ['C0041618', 'C0024485', 'C0040405', 'C1699633', 'C0002978']
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
        # modalities = ['C0041618', 'C0024485', 'C0040405', 'C1699633', 'C0002978']
        with open(file_name, 'r') as f:
            if skip_head:
                next(f)
            for line in f:
                image = line.replace('\n', '').split(split_token)
                concepts = image[1].split(';')
                # concepts = [x for x in concepts if x not in modalities]  # remove modalities
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
                if training_data[img]:
                    tags.extend(training_data[img].split(';'))
            else:
                tags.extend(training_data[img])
        tags = set(tags)
        return list(tags)

    def load_encoder(self, path):
        if isinstance(path, str):
            loaded_model = tf.keras.models.load_model(path, compile=False, custom_objects={'GeM': utils.GeM})

            self.encoder = tf.keras.models.Model(inputs=loaded_model.input, 
                                                 outputs=loaded_model.get_layer('gem_pool').output)
        else:
            self.encoder = path
        
        print(self.encoder.summary())

    
    def get_embeddings(self, data):
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
            embedding = self.encoder.predict(img_array, verbose=0).T.flatten()
            embedding = embedding / np.linalg.norm(embedding)
            embeddings[img_name] = embedding
        return embeddings
    
    def calculate_similarities(self, training_embeddings_array):
        similarities = dict()  # keys -> validation images.
        for val_img in tqdm(self.val_data):
            sims = training_embeddings_array @ self.validation_embeddings[val_img].flatten()  # normed embeddings else dot
            sims = np.array(sims).flatten()
            similarities[val_img] = sims
        # pickle.dump(similarities, open('knn_similarities.pkl', 'wb'))
        return similarities
    
    def train_tune_test(self):

        self.init_structures(skip_head=self.configuration['data']['skip_head'], 
                             split_token=self.configuration['data']['split_token'])

        self.load_encoder(self.backbone)

        print('\nCalculating training images\' embeddings...')
        # self.training_embeddings = self.get_embeddings(self.train_data)
        if self.load_embeddings:
            self.training_embeddings = pickle.load(open(self.configuration['embeddings']['train_path'], 'rb'))
            new_embeddings_dict_train = dict()
            for image_name, embedding in self.training_embeddings.items():
                new_image_name = image_name + '.jpg'
                new_embeddings_dict_train[new_image_name] = embedding.T.flatten() / np.linalg.norm(embedding.T.flatten())
            self.training_embeddings = dict(new_embeddings_dict_train)
        else:
            self.training_embeddings = self.get_embeddings(self.train_data)
        print('\nGot training images\' embeddings!')
        print('\nCalculating validation images\' embeddings...')
        # self.validation_embeddings = self.get_embeddings(self.val_data)
        if self.load_embeddings:
            self.validation_embeddings = pickle.load(open(self.configuration['embeddings']['val_path'], 'rb'))
            new_embeddings_dict_val = dict()
            for image_name, embedding in self.validation_embeddings.items():
                new_image_name = image_name + '.jpg'
                new_embeddings_dict_val[new_image_name] = embedding.T.flatten() / np.linalg.norm(embedding.T.flatten())
            self.validation_embeddings = dict(new_embeddings_dict_val)
        else:
            self.validation_embeddings = self.get_embeddings(self.val_data)
        print('\nGot validation images\' embeddings!')
        print('\nCalculating test images\' embeddings...')
        # self.test_embeddings = self.get_embeddings(self.test_data)
        if self.load_embeddings:
            self.test_embeddings = pickle.load(open(self.configuration['embeddings']['test_path'], 'rb'))
            new_embeddings_dict_test = dict()
            for image_name, embedding in self.test_embeddings.items():
                new_image_name = image_name + '.jpg'
                new_embeddings_dict_test[new_image_name] = embedding.T.flatten() / np.linalg.norm(embedding.T.flatten())
            self.test_embeddings = dict(new_embeddings_dict_test)
        else:
            self.test_embeddings = self.get_embeddings(self.test_data)
        print('\nGot test images\' embeddings!')

        with open('/home/Sainte-Genevieve/imageclef2024/embeddings_foivos/embedding_dict_test2024.pkl', 'rb') as f:
            self.off_test_embeddings = pickle.load(f)
        new_embeddings_dict_off_test = dict()
        for image_name, embedding in self.off_test_embeddings.items():
            new_image_name = image_name
            new_embeddings_dict_off_test[new_image_name] = embedding.T.flatten() / np.linalg.norm(embedding.T.flatten())
        self.off_test_embeddings = dict(new_embeddings_dict_off_test)  # has .jpg
        # print(self.off_test_embeddings)

        self.ids = [i for i in self.training_embeddings]  # img ids
        # self.ids = np.array(self.ids)
        training_embeddings_array = np.array([self.training_embeddings[i] for i in self.training_embeddings])
        print(training_embeddings_array.shape)
        # training_embeddings_array = np.array([self.training_embeddings[i] / np.linalg.norm(self.training_embeddings[i]) for i in self.training_embeddings])

        print('\nCalculating similarity between validation and training images...')
        images_sims = self.calculate_similarities(training_embeddings_array, )
        if self.weighted:
            # best_k, best_t, val_score = self.tune_weighted(similarities=images_sims)
            best_k, best_t, best_weights, val_score = self.tune_weighted_learnable(similarities=images_sims)
            # test_score, test_results = self.test_weighted(best_k=best_k, best_t=best_t, 
            #                                               training_embeddings_array=training_embeddings_array)
            test_score, test_results = self.test_weighted_learnable(best_k=best_k, best_t=best_t, 
                                                                    best_weights=best_weights,
                                                                    training_embeddings_array=training_embeddings_array)
            self.off_test_weighted_learnable(best_k=best_k, best_t=best_t, best_weights=best_weights, 
                                             training_embeddings_array=training_embeddings_array)
        else:
            best_k, best_r, val_score = self.tune(similarities=images_sims)
            test_score, test_results = self.test(best_k=best_k, best_r=best_r,
                                                 training_embeddings_array=training_embeddings_array)
        s = ('Development score = ' + str(test_score) +
             ' with k = ' + str(best_k) + ' and validation score = ' + str(val_score))
        
        print(s)

        return test_score, test_results
            


    def run(self):

        test_score, test_results = self.train_tune_test()

        if self.configuration.get('save_results'):
            print('\n\nSaving results...\n')
            with open(self.configuration.get('results_path'), 'w') as out_test:
                for result in test_results:
                    out_test.write(result + '\t' + test_results[result] + '\n')
            print('Results saved!')
    
    def tune(self, similarities):
        best_k = -1
        best_r = -1
        max_score = -1
        print('\nSorting similarities...')
        sorted_sims = dict()
        # print(similarities)
        for val_image in similarities:
            # indices.
            sorted_sims[val_image] = np.argsort(similarities[val_image])
        # print(sorted_sims)
        print('\nSimilarities sorted.')
        print('\nTuning k and r...\n')
        rs = list(range(self.r_range[0], self.r_range[1] + 1))
        for k in tqdm(range(self.k_range[0], self.k_range[1] + 1, self.k_step)):
            y_pred_val = dict()
            for r in rs:
                y_pred_val[r] = dict()
            
            # count_print = 0
            for val_image in self.val_data:
                tags_list = list()
                # tags_sum = 0
                # indices of top k highest similarities.
                top_k = sorted_sims[val_image][-k:]
                # sim = similarities[val_image][top_k]  # top k highest similarities.
                # weights = sim / sum(sim)  # normalizing by dividing with sum.
                # weighted_tags = 0  # weighted r parameter.
                for i, index in enumerate(top_k):
                    # print(index, type(index))
                    tags = self.train_data[self.ids[index]].split(';')
                    # tags_sum += len(tags)  # for averaging.
                    # weighting formula calculation.
                    # weighted_tags += weights[i] * len(tags)
                    # tags_list.extend(tags)
                    for tag in tags:
                        tags_list.append(tag)
                for r_ in rs:
                    r = r_
                    # if r == 'average':
                        # r = round(tags_sum / k)
                    # if r == 'weighting':
                        # r = int(round(weighted_tags))
                    # finding r most common tags.
                    most_frequent_tags = Counter(tags_list).most_common(r)
                    # taking only tag's name.
                    y_pred_val[r_][val_image] = ';'.join(t[0] for t in most_frequent_tags)
            for r in rs:
                f1_score = utils.evaluate_f1(self.val_data, y_pred_val[r], test=False)
                print(f'k = {k}, r = {r} - f1 = {f1_score}')
                if f1_score > max_score:
                    max_score = f1_score
                    best_k = k
                    best_r = r
        print('The best F1 score on validation data ' +
              ' is ' + str(max_score) +
              ' achieved with k = ' + str(best_k) + ' and r = ' + str(best_r) + '\n')
        return best_k, best_r, max_score
    
    def tune_weighted(self, similarities):
        best_k = -1
        best_t = -1.
        max_score = -1
        print('\nSorting similarities...')
        sorted_sims = dict()
        # print(similarities)
        for val_image in similarities:
            # indices.
            sorted_sims[val_image] = np.argsort(similarities[val_image])
        # print(sorted_sims)
        print('\nSimilarities sorted.')
        print('\nTuning k...\n')

        ts = np.arange(0.1, 1., 0.05)

        for k in tqdm(range(self.k_range[0], self.k_range[1] + 1, self.k_step)):
            if k == 0:
                k = 1
            y_pred_val = dict()
            for t in ts:
                y_pred_val[t] = dict()

            linear_weights = np.array(list(reversed(range(1, k + 1))))
            # linear_weights = list(tf.nn.softmax([float(w) for w in linear_weights]).numpy())
            if k != 1:
                linear_weights = (linear_weights - min(linear_weights)) / (max(linear_weights) - min(linear_weights))  # scale to [0., 1.]
                linear_weights = linear_weights / linear_weights.sum()  # sum to 1
            for val_image in self.val_data:

                top_k = sorted_sims[val_image][-k:]
                top_k = np.flip(top_k)  # largest to smallest
                
                f = np.zeros(shape=len(self.tags_list))
                for i, nn_index in enumerate(top_k):
                    nn_tags = self.train_data[self.ids[nn_index]].split(';')
                    presence = np.array([int(t in nn_tags) for t in self.tags_list])

                    f = f + (linear_weights[i] * presence)
                
                f = f / sum(linear_weights)

                for t in ts:
                    tags_list = list()
                    for i in np.argwhere(f >= t).flatten():
                        tags_list.append(self.tags_list[i])

                    y_pred_val[t][val_image] = ';'.join(tags_list)
            
            
            for t in ts:
                f1_score = utils.evaluate_f1(self.val_data, y_pred_val[t], test=False)
                print(f'k = {k}, t = {t} - f1 = {f1_score}')
                if f1_score > max_score:
                    max_score = f1_score
                    best_k = k
                    best_t = t
        
        print('The best F1 score on validation data ' +
              ' is ' + str(max_score) +
              ' achieved with k = ' + str(best_k) + ' and t = ' + str(best_t) + '\n')
        return best_k, best_t, max_score
    
    def tune_weighted_learnable(self, similarities):
        from knn_weights_learner import GeneticWeightsSearch
        best_k = -1
        best_t = -1.
        max_score = -1
        best_weights = None
        print('\nSorting similarities...')
        sorted_sims = dict()
        # print(similarities)
        for val_image in similarities:
            # indices.
            sorted_sims[val_image] = np.argsort(similarities[val_image])
        # print(sorted_sims)
        print('\nSimilarities sorted.')
        print('\nTuning k...\n')

        # ts = np.arange(0.1, 1., 0.05)
        ts = [0.35]

        for k in tqdm([50]):
            for t in ts:
                if k == 0: continue
                print('\nRunning weights optimization...')
                searcher = GeneticWeightsSearch(k=k, threshold=t, train_data=self.train_data, 
                                                val_data=self.val_data, tags=self.tags_list, 
                                                ids=self.ids, sorted_sims=sorted_sims)
                weights, f1_score = searcher.optimize_weights(population_size=200, num_generations=5, 
                                                        crossover_rate=0.95, mutation_rate=0.1)
                if f1_score > max_score:
                    max_score = f1_score
                    best_k = k
                    best_t = t
                    best_weights = weights
        
        print('The best F1 score on validation data ' +
              ' is ' + str(max_score) +
              ' achieved with k = ' + str(best_k) + ' and t = ' + str(best_t) + '\n')
        return best_k, best_t, best_weights, max_score
    
    def test(self, best_k, best_r, training_embeddings_array):
        print('\nStarting testing...\n')
        y_pred_test = dict()
        for test_img in tqdm(self.test_data):

            sims = training_embeddings_array @ self.test_embeddings[test_img].flatten()
            sims = np.array(sims).flatten()

            # indices of top k highest similarities.
            top_k = np.argsort(sims)[-best_k:]
            tags_list = list()
            tags_sum = 0
            sim = sims[top_k]  # top k highest similarities.
            weights = sim / sum(sim)
            weighted_tags = 0  # weighted r parameter.
            for i, index in enumerate(top_k):
                tags = self.train_data[self.ids[index]].split(';')
                tags_sum += len(tags)
                weighted_tags += weights[i] * len(tags)
                for tag in tags:
                    tags_list.append(tag)
            r = best_r
            if r == 'average':
                r = round(tags_sum / best_k)
            if r == 'weighting':
                r = int(round(weighted_tags))
            # finding r most common tags.
            most_frequent_tags = Counter(tags_list).most_common(r)
            # print(most_frequent_tags)
            # taking only tag's name.
            y_pred_test[test_img] = ';'.join(t[0] for t in most_frequent_tags)

        # sim_file.close()
        f1_score, p, r, _ = utils.evaluate_f1(self.test_data, y_pred_test, test=True)
        print('\nThe F1 score on dev set is: {}\n'.format(f1_score))
        print('Precision score:', p)
        print('Recall score:', r)
        return f1_score, y_pred_test

    def test_weighted(self, best_k, best_t, training_embeddings_array):
        print('\nStarting testing...')
        y_pred_test = dict()
        for test_img in tqdm(self.test_data):
            sims = training_embeddings_array @ self.test_embeddings[test_img].flatten()
            sims = np.array(sims).flatten()

            tags_list = list()
            # indices of top k highest similarities.
            top_k = np.argsort(sims)[-best_k:]
            top_k = np.flip(top_k)  # largest to smallest (descending)

            linear_weights = np.array(list(reversed(range(1, best_k + 1))))
            if best_k != 1:
                linear_weights = (linear_weights - min(linear_weights)) / (max(linear_weights) - min(linear_weights))  # scale to [0., 1.]
                linear_weights = linear_weights / linear_weights.sum()  # sum to 1.
            f = np.zeros(shape=len(self.tags_list))

            for i, nn_index in enumerate(top_k):
                nn_tags = self.train_data[self.ids[nn_index]].split(';')
                presence = np.array([int(t in nn_tags) for t in self.tags_list])

                f = f + (linear_weights[i] * presence)

            
            f = f / sum(linear_weights)

            for i in np.argwhere(f >= best_t).flatten():
                tags_list.append(self.tags_list[i])

            y_pred_test[test_img] = ';'.join(tags_list)
        
        f1_score, p, r, _ = utils.evaluate_f1(self.test_data, y_pred_test, test=True)
        print('\nThe F1 score on dev set is: {}\n'.format(f1_score))
        print('Precision score:', p)
        print('Recall score:', r)
        return f1_score, y_pred_test
    
    def test_weighted_learnable(self, best_k, best_t, best_weights,
                                training_embeddings_array):
        print('\nStarting testing...')
        y_pred_test = dict()
        for test_img in tqdm(self.test_data):
            sims = training_embeddings_array @ self.test_embeddings[test_img].flatten()
            sims = np.array(sims).flatten()

            tags_list = list()
            # indices of top k highest similarities.
            top_k = np.argsort(sims)[-best_k:]
            top_k = np.flip(top_k)  # largest to smallest (descending)

            f = np.zeros(shape=len(self.tags_list))

            for i, nn_index in enumerate(top_k):
                nn_tags = self.train_data[self.ids[nn_index]].split(';')
                presence = np.array([int(t in nn_tags) for t in self.tags_list])

                f = f + (best_weights[i] * presence)

            
            f = f / sum(best_weights)

            for i in np.argwhere(f >= best_t).flatten():
                tags_list.append(self.tags_list[i])

            y_pred_test[test_img] = ';'.join(tags_list)
        
        f1_score, p, r, _ = utils.evaluate_f1(self.test_data, y_pred_test, test=True)
        print('\nThe F1 score on dev set is: {}\n'.format(f1_score))
        print('Precision score:', p)
        print('Recall score:\n', r)
        return f1_score, y_pred_test
    
    def off_test_weighted_learnable(self, best_k, best_t, best_weights,
                                    training_embeddings_array):
        print('\nStarting testing...')
        y_pred_off_test = dict()
        for test_img in tqdm(self.off_test_data):
            sims = training_embeddings_array @ self.off_test_embeddings[test_img].flatten()
            sims = np.array(sims).flatten()

            tags_list = list()
            # indices of top k highest similarities.
            top_k = np.argsort(sims)[-best_k:]
            top_k = np.flip(top_k)  # largest to smallest (descending)

            f = np.zeros(shape=len(self.tags_list))

            for i, nn_index in enumerate(top_k):
                nn_tags = self.train_data[self.ids[nn_index]].split(';')
                presence = np.array([int(t in nn_tags) for t in self.tags_list])

                f = f + (best_weights[i] * presence)

            
            f = f / sum(best_weights)

            for i in np.argwhere(f >= best_t).flatten():
                tags_list.append(self.tags_list[i])

            y_pred_off_test[str(test_img).split('.', 1)[0]] = ';'.join(tags_list)  # no .jpg
        
        
        print('\n\nSaving official test results...\n')
        with open('knn_learnable_weights_off_test_pred.csv', 'w') as out_test:
            for result in y_pred_off_test:
                out_test.write(result + ',' + y_pred_off_test[result] + '\n')
        print('Results saved!')

        return y_pred_off_test

        