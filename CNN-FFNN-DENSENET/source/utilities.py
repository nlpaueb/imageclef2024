import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow_addons.utils.keras_utils import LossFunctionWrapper


class GeM(tf.keras.layers.Layer):
    """
    GeM global pooling layer
    Based on: Fine-tuning CNN Image Retrieval with No Human Annotation (https://arxiv.org/pdf/1711.02512.pdf)
    """
    def __init__(self, data_format='channels_last', init_norm=3.0, normalize=False, **kwargs):
        self.p = None
        self.init_norm = init_norm
        self.normalize = normalize
        self.data_format = data_format

        super(GeM, self).__init__(**kwargs)

    def build(self, input_shape):
        feature_size = input_shape[-1] if self.data_format == 'channels_last' else input_shape[1]
        self.p = self.add_weight(name='norms', shape=(feature_size,),
                                 initializer=tf.keras.initializers.constant(self.init_norm),
                                 trainable=True, constraint=tf.keras.constraints.NonNeg())
        print('p shape:', self.p.shape)
        super(GeM, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'normalize': self.normalize,
            'data_format': self.data_format
        })
        return config

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        x = tf.clip_by_value(x, clip_value_min=1e-6, clip_value_max=tf.reduce_max(x))
        if self.data_format == 'channels_last':
            x = tf.math.pow(x, self.p)
            x = tf.math.reduce_mean(x, axis=[1, 2] if len(inputs.shape) == 4 else 1, keepdims=False)
            # x = tf.math.pow(x, 1.0 / self.p)
        else:
            x = tf.math.pow(x, tf.expand_dims(tf.expand_dims(self.p, axis=1), axis=1))
            x = tf.math.reduce_mean(x, axis=[2, 3] if len(inputs.shape) == 4 else 2, keepdims=False)
            # x = tf.squeeze(x, axis=[1, 2])
        
        x = tf.math.pow(x, 1.0 / self.p)

        if self.normalize:
            x = tf.math.l2_normalize(x, axis=1)
        return x

    def compute_output_shape(self, input_shape):
        return tuple([None, input_shape[-1]])


class ReturnBestEarlyStopping(tf.keras.callbacks.EarlyStopping):
    """
    Early Stopping class that restores the weights of the best epoch
    """
    def __init__(self, **kwargs):
        super(ReturnBestEarlyStopping, self).__init__(**kwargs)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            if self.verbose > 0:
                print(f'\nEpoch {self.stopped_epoch + 1}: early stopping')
        elif self.restore_best_weights:
            if self.verbose > 0:
                print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)


def loss_1_minus_f1_examples(y_true, y_pred):
    # average over the examples, not labels...
    tp = tf.keras.backend.sum(tf.keras.backend.cast(y_true * y_pred, 'float'), axis=-1)
    fp = tf.keras.backend.sum(tf.keras.backend.cast((1 - y_true) * y_pred, 'float'), axis=-1)
    fn = tf.keras.backend.sum(tf.keras.backend.cast(y_true * (1 - y_pred), 'float'), axis=-1)
    p = tp / (tp + fp + tf.keras.backend.epsilon())
    r = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    # print(f1.shape)  # batch size
    # print((1 - tf.keras.backend.mean(f1)).shape)  # scalar
    # print((1 - f1).shape)  # batch size

    return 1 - tf.keras.backend.mean(f1)


def loss_1_minus_f1_labels(y_true, y_pred):
    # average over the labels
    tp = tf.keras.backend.sum(tf.keras.backend.cast(y_true * y_pred, 'float'), axis=0)
    fp = tf.keras.backend.sum(tf.keras.backend.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = tf.keras.backend.sum(tf.keras.backend.cast(y_true * (1 - y_pred), 'float'), axis=0)
    p = tp / (tp + fp + tf.keras.backend.epsilon())
    r = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    # print(f1.shape)  # labels size
    return 1 - tf.keras.backend.mean(f1)


def loss_1mf1_by_bce(y_true, y_pred):
    """
    returns f1 loss * bce
    :param y_true: gold truth vector of the batch
    :param y_pred: predicted vector of the batch
    :return: loss value
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # print('Y shapes:', y_true.shape, y_pred.shape)
    loss_f1 = loss_1_minus_f1_examples(y_true, y_pred)
    # loss_f1 = samples_double_soft_f1(y_true, y_pred)
    bce = tf.keras.metrics.binary_crossentropy(y_true=y_true, y_pred=y_pred,
                                               from_logits=False)  # size = batch_size
    # bce = tf.keras.backend.binary_crossentropy(target=y_true, output=y_pred,
    #                                            from_logits=False)
    # print(tf.keras.backend.mean(loss_f1 * bce, axis=1))
    # print(loss_f1 * bce)
    # print(tf.reduce_mean(loss_f1 * bce, axis=-1))
    # tf.debugging.assert_equal(tf.keras.backend.mean(loss_f1 * bce, axis=1), loss_f1 * bce)
    # print(loss_f1, bce)
    return loss_f1 * bce  # batch size


class AsymmetricLoss(LossFunctionWrapper):
    """
    ASL loss: Asymmetric Loss For Multi-Label Classification (https://arxiv.org/abs/2009.14119)
    Implementation: https://github.com/SmilingWolf/SW-CV-ModelZoo/blob/main/Losses/ASL.py
    """
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=0,
        clip=0.05,
        eps=1e-7,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="asymmetric_loss",
    ):
        super().__init__(
            asymmetric_loss,
            name=name,
            reduction=reduction,
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            eps=eps,
        )


@tf.function
def asymmetric_loss(y_true, y_pred, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-7):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    clip = tf.cast(clip, dtype=tf.float32)

    xs_pos = y_pred
    xs_neg = 1 - y_pred

    if clip is not None and clip > 0:
        xs_neg = tf.clip_by_value(xs_neg + clip, clip_value_min=0, clip_value_max=1)

    # Basic CE calculation
    los_pos = y_true * tf.math.log(
        tf.clip_by_value(xs_pos, clip_value_min=eps, clip_value_max=1)
    )
    los_neg = (1 - y_true) * tf.math.log(
        tf.clip_by_value(xs_neg, clip_value_min=eps, clip_value_max=1)
    )
    loss = los_pos + los_neg

    # Asymmetric Focusing
    if gamma_neg > 0 or gamma_pos > 0:
        pt0 = xs_pos * y_true
        pt1 = xs_neg * (1 - y_true)
        pt = pt0 + pt1
        one_sided_gamma = gamma_pos * y_true + gamma_neg * (1 - y_true)
        one_sided_w = tf.math.pow(1 - pt, one_sided_gamma)
        loss *= one_sided_w

    return -tf.reduce_sum(loss, axis=-1)


def load_batch(ids, img_index, tags_index,
               images_path, tags_list, preprocessor, size=(224, 224, 3)):
    """
    loads a batch of data
    :param ids: indices of samples (list)
    :param img_index: index of X (dict)
    :param tags_index: index of Y (dict)
    :param images_path: path to X (string)
    :param tags_list: list of tags
    :param preprocessor: Keras object for preprocessing the images
    :param size: size of the images (tuple)
    :return: a batch of data
    """
    x_data, y_data = list(), list()

    for i in ids:
        if 'train' in img_index[i]:
            images_path = '../train/'
        else:
            images_path = '../valid/'
        image_path = os.path.join(images_path, img_index[i])
        # PIL image.
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=size)
        # img = auto_augment_policy(img)
        if hasattr(preprocessor, 'preprocess_input'):
            x = tf.keras.preprocessing.image.img_to_array(img)  # NumPy array.
            x = preprocessor.preprocess_input(x)
        else:
            x = np.squeeze(preprocessor(images=img)['pixel_values'])
        x_data.append(x)

        binary_concepts = np.zeros(len(tags_list), dtype=int)
        if isinstance(tags_index[i], list):
            tags_index[i] = ';'.join(tags_index[i])
        current_concepts = tags_index[i].split(';')
        for j in range(0, len(tags_list)):
            if tags_list[j] in current_concepts:
                binary_concepts[j] = 1

        y_data.append(binary_concepts)

    # x_data = np.array(x_data)
    # x_data = rand_aug(images=x_data)
    # x_data = preprocessor.preprocess_input(x_data)
    # print(y_data)
    return np.array(x_data), np.array(y_data)


def load_test_batch(ids, img_index, images_path, preprocessor, size=(224, 224, 3)):
    """
    loads a batch of data (without labels)
    :param ids: indices of data (list)
    :param img_index: index of X (dict)
    :param images_path: path to X (string)
    :param preprocessor: Keras object for preprocessing the images
    :param size: size of the images (tuple)
    :return: a batch of X
    """
    x_data = list()

    for i in ids:
        if 'train' in img_index[i]:
            images_path = '../train/'
        elif 'valid' in img_index[i]:
            images_path = '../valid/'
        else:
            images_path = '../test/'
        image_path = os.path.join(images_path, img_index[i])
        # PIL image.
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=size)
        # image_path = os.path.join(images_path, img_index[i])
        # # PIL image.
        # img = tf.keras.preprocessing.image.load_img(
        #     image_path, target_size=size)
        if hasattr(preprocessor, 'preprocess_input'):
            x = tf.keras.preprocessing.image.img_to_array(img)  # NumPy array.
            x = preprocessor.preprocess_input(x)
        else:
            x = np.squeeze(preprocessor(images=img)['pixel_values'])
        x_data.append(x)

    return np.array(x_data)


def create_index(data):
    """
    creates the index for the data [index, sample]
    :param data: data dictionary
    :return: the indices for images and labels
    """
    img_index = dict(zip(range(len(data)), list(data)))
    tags_index = dict(zip(range(len(data)), list(data.values())))
    return img_index, tags_index


def divisor_generator(n):
    """
    finds the divisors of a number
    Implementation: https://stackoverflow.com/questions/171765/what-is-the-best-way-to-get-all-the-divisors-of-a-number
    :param n: the number
    :return: yields a divisor
    """
    large_divisors = []
    for i in range(1, int(np.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor


def evaluate_f1(gt_pairs, candidate_pairs, targets=None, test=False, report_name=None):
    """
    function that computes F1 score ('samples' average)
    Implementation based on the ImageCLEF 2022 campaign (https://www.imageclef.org/2022/medical/caption)
    :param gt_pairs: dictionary of truth data
    :param candidate_pairs: dictionary of predictions
    :param targets: list of tags (optional)
    :param test: flag for testing (boolean)
    :param report_name: name for scikit-learn classification report (optional)
    :return: the F1 score (and additional scores optionally)
    """
    # Concept stats
    min_concepts = sys.maxsize
    max_concepts = 0
    total_concepts = 0
    concepts_distrib = {}

    # Define max score and current score
    max_score = len(gt_pairs)
    current_f1 = 0
    current_r = 0
    current_p = 0

    # Check there are the same number of pairs between candidate and ground truth
    if len(candidate_pairs) != len(gt_pairs):
        print('ERROR : Candidate does not contain the same number of entries as the ground truth!')
        exit(1)

    # Evaluate each candidate concept list against the ground truth
    # print('Processing concept sets...\n********************************')

    y_true_all, y_pred_all = list(), list()
    total_predicted_concepts = set()
    # recalls = open('recalls.txt', 'w')
    # if split is not None:
    #     sim_file = open(
    #         'saved files/oneNN dev similarities and score split ' + str(split) + '.txt', 'w')
    for image_key in candidate_pairs:
        # Get candidate and GT concepts
        candidate_concepts = candidate_pairs[image_key].upper()
        gt_concepts = gt_pairs[image_key]

        if isinstance(gt_concepts, list):
            gt_concepts = ';'.join(gt_concepts)
            gt_concepts = gt_concepts.upper()
        else:
            gt_concepts = gt_concepts.upper()

        # Split concept string into concept array
        # Manage empty concept lists
        if gt_concepts.strip() == '':
            gt_concepts = []
        else:
            gt_concepts = gt_concepts.split(';')

        if candidate_concepts.strip() == '':
            candidate_concepts = []
        else:
            candidate_concepts = candidate_concepts.split(';')

        # Manage empty GT concepts (ignore in evaluation)
        f1score, p, r = 0, 0, 0
        if len(gt_concepts) == 0:
            max_score -= 1
            # Normal evaluation
            if len(candidate_concepts) == 0:
                f1score = 1
                p = 1
                r = 1
                max_score += 1
        else:

            y_true_all.append(set(gt_concepts))
            y_pred_all.append(set(candidate_concepts))

            total_predicted_concepts.update(set(candidate_concepts))

            # Concepts stats
            total_concepts += len(gt_concepts)

            # Global set of concepts
            all_concepts = sorted(list(set(gt_concepts + candidate_concepts)))

            # Calculate F1 score for the current concepts
            y_true = [int(concept in gt_concepts) for concept in all_concepts]
            y_pred = [int(concept in candidate_concepts)
                      for concept in all_concepts]

            # y_true_all.append(y_true)
            # y_pred_all.append(y_pred)

            f1score = f1_score(y_true, y_pred, average='binary', zero_division=1)
            p, r = 0, 0
            if test:
                p = precision_score(y_true, y_pred, average='binary', zero_division=1)
                r = recall_score(y_true, y_pred, average='binary', zero_division=1)
            # if split is not None:
            #     sim_file.write('Cosine similarity of dev image with the 1st most similar training image: ' +
            #                    str(similarities[image_key]) + ' and F1 score: ' + '(' + str(f1score) + ')' + ' ' +
            #                    str(gt_concepts) + ' ' + str(candidate_concepts) + '\n')

            # recalls.write(str(set(gt_concepts)) + ' ' + str(set(candidate_concepts)) + ' ' + str(r) + '\n')

        # Increase calculated score
        current_f1 += f1score
        if test:
            current_p += p
            current_r += r

        # Concepts stats
        nb_concepts = str(len(gt_concepts))
        if nb_concepts not in concepts_distrib:
            concepts_distrib[nb_concepts] = 1
        else:
            concepts_distrib[nb_concepts] += 1

        if len(gt_concepts) > max_concepts:
            max_concepts = len(gt_concepts)

        if len(gt_concepts) < min_concepts:
            min_concepts = len(gt_concepts)

    # recalls.close()
    mean_f1_score = current_f1 / max_score  # averaging over images.
    if test:
        mean_p_score = current_p / max_score
        mean_r_score = current_r / max_score
    # if split is not None:
    #     sim_file.close()
    if targets is not None:
        # textfile = open("a_file.txt", "w")
        # for element in y_true_all:
        #     textfile.write(str(element) + "\n")
        # textfile.close()
        mlb = MultiLabelBinarizer()
        mlb.fit([targets])
        report = classification_report(mlb.transform(y_true_all), mlb.transform(y_pred_all),
                                       target_names=[str(cls)
                                                     for cls in mlb.classes_],
                                       zero_division=0, output_dict=True)
        clf_report = pd.DataFrame(report).transpose()
        clf_report.to_csv(report_name, index=True)
    # mean_p = current_p / max_score
    # mean_r = current_r / max_score

    if test:
        print('Total predicted concepts:', len(total_predicted_concepts))
        return mean_f1_score, mean_p_score, mean_r_score, total_predicted_concepts
    return mean_f1_score
