import numpy as np
import pandas as pd
import pickle
from scipy import stats
from tqdm import tqdm
from numpy.linalg import norm
import math
import torch 

class DMM:

    def __init__(self, hist_train_file:str, mmc_sim_file:str):
        """ The DMM sequence scoring method that we use in order to more efficiently rerank the generated sequences during beam searching.
        An illustration of the algorithm is provided in my Thesis paper.

        Args:
            hist_train_file (str): The name of the pickle file that contains the training maximum cosine similarity (between tags and relative captions) histogram.
            mmc_sim_file (str): The name of the pickle file that contains the median maximum cosine similarity value for each tag (based on calculations on the training dataset).
        """

        self.hist_train_file = hist_train_file
        self.mmc_sim_file = mmc_sim_file

        self.hist_train = self.pickle_to_dict(self.hist_train_file)
        self.mmc_sim = self.pickle_to_dict(self.mmc_sim_file)
        self.word_index = self.pickle_to_dict("/home/pkaliosis/dmm_stats_24/word_index.pkl")
        self.embedding_matrix = np.load("/home/pkaliosis/dmm_stats_24/embedding_matrix.npy")
        self.respective_tags = list()
        self.centroid_embeddings, self.gen_tags_dict = dict(), dict()

        max_len = self.embedding_matrix.shape[0]
        self.word_index['startsequence'] = max_len + 1
        self.word_index['endsequence'] = max_len + 2
        self.word_index['<unk>'] = max_len + 3
        self.word_index['endofsequence'] = max_len + 4

        #print('mmc:', self.mmc_sim)
        self.mmc_sim['Pneumopericardium'] = [0.85]

        #print(self.word_index)

        for i in range(5):
            to_add = np.array([np.ones(300)])
            #print('to add size:', to_add.size)
            self.embedding_matrix = np.append(self.embedding_matrix, to_add, axis=0)
            #self.embedding_matrix.append(np.ones(300))

        new_len = self.embedding_matrix.shape[0]

        concepts_mapper = pd.read_csv('/home/pkaliosis/cnn_rnn/ImageCLEFmedical_Caption_2023_cui_mapping.csv', sep="\t", header=None, names=['cui', 'concept'])

        # Build a mapper
        self._concepts_dict = {}
        for row in concepts_mapper['concept']:
            mapper = concepts_mapper.loc[concepts_mapper['concept'] == row].values.flatten().tolist()
            self._concepts_dict[mapper[0]] = mapper[1]

        


    def pickle_to_dict(self, file):

        # the hist_train pkl file will essentially contain the train histograms for each tag
        # it is calculated as: for each caption that comprises the tag, retrieve the max cosine similarity between the tag and the caption's words!
        # save dictionary to pickle file
        file_to_read = open(file, "rb")
        loaded_hist = pickle.load(file_to_read)

        return loaded_hist

    
    # Define function that calculates the given text's word embeddings centroid.
    def text_centroid(self, text, model, word_index):
        """ Calculate centroid function """
        text_vec =[]
        counter = 0
        text = text.split(" ")
        for word in text:
            try:
                if (counter == 0):
                    text_vec = model[word_index[word.lower()]]
                    counter+=1
                else:
                    text_vec = np.add(text_vec, model[word_index[word.lower()]])
                    counter+=1
            except:
                pass

        return np.asarray(text_vec) / counter

    
    # Define function that calculates the word embeddings of each item in the given list
    def get_concept_word_embeddings(self, _concepts:list, dims):

        concepts_embeddings = list()
        if dims == 2:
            for i, clist in enumerate(_concepts):
                concepts_embeddings.append([])
            for c in clist:
                c = c.replace('-', ' ')
                c = c.replace('.', ' ')
                c = c.replace(':', ' ')
                c = c.replace('[', ' ')
                c = c.replace(']', ' ')
                c = c.replace('(', ' ')
                c = c.replace(')', ' ')
                c = c.replace('=', ' ')
                c = c.replace('/', ' ')

                if ((len(c.split(' ')) == 1)):
                    # if tag is only one word --> word_embedding(tag)
                    if c.lower() in self.word_index:
                        #print(self.embedding_matrix[self.word_index[c.lower()]])
                        concepts_embeddings[i].append(self.embedding_matrix[self.word_index[c.lower()]])
                    else:
                        concepts_embeddings[i].append(np.zeros(300))
                else:
                    # else if tag is more than one word --> centroid of words embeddings of each tag subword
                    if c not in self.centroid_embeddings.keys():
                        centroid_emb = self.text_centroid(c, self.embedding_matrix, self.word_index)
                        concepts_embeddings[i].append(centroid_emb)
                        self.centroid_embeddings[c] = centroid_emb
                    else:
                        concepts_embeddings[i].append(self.centroid_embeddings[c])

        elif dims == 1:

            for i, c in enumerate(_concepts):

                key = c

                c = c.replace('-', ' ')
                c = c.replace('.', ' ')
                c = c.replace(':', ' ')
                c = c.replace('[', ' ')
                c = c.replace(']', ' ')
                c = c.replace('(', ' ')
                c = c.replace(')', ' ')
                c = c.replace('=', ' ')
                c = c.replace('/', ' ')

                if ((len(c.split(' ')) == 1)):
                    # if tag is only one word --> word_embedding(tag)
                    #print('c lower:', c.lower())
                    if c.lower() in self.word_index:
                        #print('found in word index')
                        #print(embedding_matrix[word_index[c.lower()]])
                        #print('word id:', self.word_index[c.lower()])
                        #print('embedding matrix...:', self.embedding_matrix[self.word_index[c.lower()]])
                        concepts_embeddings.append(self.embedding_matrix[self.word_index[c.lower()]])
                        self.respective_tags.append(key)
                        #print('added embeddings:', concepts_embeddings[i])
                    else:
                        #print('not found in word index!')
                        concepts_embeddings.append(np.zeros(300))
                else:
                    # else if tag is more than one word --> centroid of words embeddings of each tag subword
                    #concepts_embeddings.append(self.text_centroid(c, self.embedding_matrix, self.word_index))
                    #self.respective_tags.append(key)
                    if c not in self.centroid_embeddings.keys():
                        centroid_emb = self.text_centroid(c, self.embedding_matrix, self.word_index)
                        concepts_embeddings.append(centroid_emb)
                        self.respective_tags.append(key)
                        self.centroid_embeddings[c] = centroid_emb
                    else:
                        concepts_embeddings.append(self.centroid_embeddings[c])
                        self.respective_tags.append(key)

        return concepts_embeddings


    
    def get_captions_word_embeddings(self, _captions:list, dims):

        captions_embeddings = list()

        if dims == 2:
            for i, clist in enumerate(_captions):
                captions_embeddings.append([])
            for c in clist.split(' '):
                c = c.replace('-', ' ')
                c = c.replace('.', ' ')
                c = c.replace(':', ' ')
                c = c.replace('[', ' ')
                c = c.replace(']', ' ')
                c = c.replace('(', ' ')
                c = c.replace(')', ' ')
                c = c.replace('=', ' ')
                c = c.replace('/', ' ')


                if ((len(c.split(' ')) == 1)):
                    if c.lower() in self.word_index:
                        captions_embeddings[i].append(self.embedding_matrix[self.word_index[c.lower()]])
                    else:
                        captions_embeddings[i].append(np.zeros(300))
                elif ((len(c.split()) > 1) and (len(self.text_centroid(c, self.embedding_matrix, self.word_index)) > 0)):
                    captions_embeddings[i].append(self.text_centroid(c, self.embedding_matrix, self.word_index))
                else:
                    captions_embeddings[i].append(np.zeros(300))

        elif dims == 1:
            for i, c in enumerate(_captions):
                c = c.replace('-', ' ')
                c = c.replace('.', ' ')
                c = c.replace(':', ' ')
                c = c.replace('[', ' ')
                c = c.replace(']', ' ')
                c = c.replace('(', ' ')
                c = c.replace(')', ' ')
                c = c.replace('=', ' ')
                c = c.replace('/', ' ')

                caption_centroid = self.text_centroid(c, self.embedding_matrix, self.word_index)

                if ((len(c.split(' ')) == 1)):
                    #print('c:', c)
                    #print('word index:', self.word_index[c.lower()])
                    if c.lower() in self.word_index:
                        captions_embeddings.append(self.embedding_matrix[self.word_index[c.lower()]])
                    else:
                        captions_embeddings.append(np.zeros(300))
                elif ((len(c.split()) > 1) and (len(caption_centroid) > 0)):
                    captions_embeddings.append(caption_centroid)
                else:
                    captions_embeddings.append(np.zeros(300))


        return captions_embeddings


    def compute_sims(self, concepts_embeds:list, captions_embeds:list, concepts, flag=False):
        #print('concepts in compute sims:', concepts)
        similarities = list()

        if (isinstance(concepts, str)):
            concepts = [concepts]

        for i, tags_i in enumerate(concepts):
            similarities.append([])
            for k in range(len(captions_embeds)):
                #print('concept embeds i:', concepts_embeds[i])
                #print('len concepts embeds:', (concepts_embeds[i].size))
                #print('concepts embeds 0:', (concepts_embeds[i]))
                #print('len caption embeds:', (captions_embeds[k].size))
                if flag:
                    similarities[i].append(self.cosine_sim(concepts_embeds, captions_embeds[k]))
                else:
                    similarities[i].append(self.cosine_sim(concepts_embeds[i], captions_embeds[k]))

        return similarities


    def init_hist(self, concepts):
        gen_tags_dict = dict()

        for c in concepts:
            tags = c.split(';')

        for t in tags:
            if t not in gen_tags_dict.keys():
                gen_tags_dict[t] = list()

        return gen_tags_dict

    # compute cosine similarity
    def cosine_sim(self, A, B):
        #print('np dot a b:', np.dot(A, B))
        #print('norm A:', norm(A))
        #print('norm B:', norm(B))
        #print('---------------------------------------')

        norm_a = norm(A)
        norm_b = norm(B)

        """if norm(A) == 0:
            norm_a = 1
        else:
            norm_a = norm(A)

        if norm(B) == 0:
            norm_b = 1
        else:
            norm_b = norm(B)

        """
        cosine = np.dot(A,B)/(norm_a*norm_b)
        return cosine

    
    def compute_hist(self, concepts_embeddings, captions_embeddings, gen_tags_dict, concepts):
        # iterate through the dataset captions
        #for i in tqdm(range(len(captions_embeddings))):
            #for j in tqdm(range(len(concepts_embeddings))):
            # for each caption compute the cosine similarity between each tag and each caption word
            # ie. if #tags = 2 and len(caption)=10, then a matrix of size (2, 10) is returned
        sims = self.compute_sims(concepts_embeddings, captions_embeddings, concepts, False)

        """print('sims shape:', len(sims))
        print('len sims 0:', len(sims[0]))
        print('sims 0:', sims[0])
        """

        # iterate through the sims vector
        for k, rt in enumerate(sims):
            sims[k] = [x for x in sims[k] if (math.isnan(x))==False]
            if len(sims[k]) > 0:
                #print('respective tag:', self.respective_tags[k])
                if (concepts[k] in gen_tags_dict.keys()):
                    gen_tags_dict[concepts[k]].append(np.max(sims[k]))
                else:
                    #print('in else:')
                    gen_tags_dict[concepts[k]] = list()
                    gen_tags_dict[concepts[k]].append(np.max(sims[k]))
            else:
                print('Empty sims list!!!')

        return gen_tags_dict


    def compute_histogram_divergence(self, train_hist, gen_hist):

        score = 0
        #print('tags for hist:', gen_hist.keys())
        for key in gen_hist.keys():
            train_list = list(train_hist[key])
            gen_list = list(gen_hist[key])

            #KS-test looks suitable!
            ks = stats.kstest(train_list, gen_list)
            #print('ks:', ks[0])
            score += ks[0]

        aggregated_score = score / len(gen_hist.keys())
        return aggregated_score



    def dmm_loss(self, caption_embeds, concept_embeds, concept):

        cos_t = self.compute_sims(concept_embeds, caption_embeds, concept, True)
        #print('len cos_t:', len(cos_t))
        #print('size cos_t 0:', len(cos_t[0]))
        cos_t[0] = [x for x in cos_t[0] if str(x) != 'nan']
        #print('cos_t 0:', cos_t[0])
        #print(sorted(cos_t[0], reverse=True))
        max_cos_t = np.mean(sorted(cos_t[0], reverse=True)[:10])
        #print('max cos t:', max_cos_t)
        #print('max cos t:', max_cos_t)

        if concept in self.mmc_sim.keys():
            max_cos_c = self.mmc_sim[concept][0]
        else:
            max_cos_c = 0.5
        #print('max cos c:', max_cos_c)

        dmm = (max_cos_t - max_cos_c) ** 2
        return dmm

    def check_for_nan(self, t):
        if torch.is_tensor(t):
            """t_bool = torch.isnan(t)
            flag = True
            for b in t_bool:
                if b == True:
                    flag = False
            
            return flag"""
            return True
        else:
            if t == '':
                return True
            else:
                return False

    
    def dmm_handler(self, caption, concepts, calc_dmm_loss=True):
        """For a given caption and each assigned concepts, calculate the dmm loss, as defined in my MSc Thesis.
        
        Args:
            caption: The caption (in text format) for which we want to calculate the dmm loss.
            concepts: A list of the concepts (in text format) that are assigned to the given caption.
        """

        #init_hist = self.init_hist(concepts)

        if calc_dmm_loss:

            #concepts = concepts.split(';')
            if (self.check_for_nan(concepts[0])):
                concepts = []
            else:
                concepts = concepts[0].split(';')
            #print('splitted concepts:', concepts)
            for i, c in enumerate(concepts):
                #concepts[i] = self._concepts_dict[c]
                if c not in self._concepts_dict.keys():
                    c = "C0040405"
                    concepts[i] = self._concepts_dict[str(c)]
                else:
                    concepts[i] = self._concepts_dict[str(c)]

            #print('Caption:', caption)
            #print('Concepts:', concepts)

            caption_embeddings = self.get_captions_word_embeddings(caption, dims=1)
            concept_embeddings = self.get_concept_word_embeddings(concepts, dims=1)

            #self.gen_tags_dict = self.compute_hist(concept_embeddings, caption_embeddings, self.gen_tags_dict, concepts)

            #print('gen tags dict:', self.gen_tags_dict)

            # calculate KL divergence between train histogram and generated histogram
            #ks_score = self.compute_histogram_divergence(self.hist_train, self.gen_tags_dict)

            dmm_loss_sum = 0
            for i, c in enumerate(concepts):
                dmm_loss_sum += self.dmm_loss(caption_embeddings, concept_embeddings[i], concepts[i])

            return dmm_loss_sum
        
        else:

            #concepts = concepts.split(';')
            if (self.check_for_nan(concepts[0])):
                concepts = ['C0040405']
            else:
                #print('concepts:', concepts)
                concepts = concepts[0].split(';')
            #print('splitted concepts:', concepts)
            for i, c in enumerate(concepts):
                #print("conceptaki:", c)
                if c not in self._concepts_dict.keys():
                    c = "C0040405"
                    concepts[i] = self._concepts_dict[str(c)]
                else:
                    concepts[i] = self._concepts_dict[str(c)]

            #print('Caption:', caption)
            #print('Concepts:', concepts)

            caption_embeddings = self.get_captions_word_embeddings(caption, dims=1)
            concept_embeddings = self.get_concept_word_embeddings(concepts, dims=1)

            self.gen_tags_dict = self.compute_hist(concept_embeddings, caption_embeddings, self.gen_tags_dict, concepts)

            #print('gen tags dict:', self.gen_tags_dict)

            # calculate KL divergence between train histogram and generated histogram
            ks_score = self.compute_histogram_divergence(self.hist_train, self.gen_tags_dict)

            return ks_score





    