from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, AutoTokenizer, LogitsProcessorList, PhrasalConstraint
import torch
from PIL import Image
import requests

import numpy as np
import pandas as pd
import json
import copy
import argparse, os
from tqdm import tqdm

from torch.utils.data import DataLoader

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

from instructBLIPdataset_2 import CustomVisionDataset

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize

from torch.cuda.amp import autocast, GradScaler

from dmm import DMM
from dmm_logits import DMMLogits


RESULTS_PATH = '/home/pkaliosis/instructBLIP/results/submission_test_instructblip_2024_dmm_02.csv'
GENERATIONS_PATH = '/path/to/generated_captions.csv'
BEST_MODEL_PATH = '/path/to/best-model.pt'

class InstructBLIP:

    def __init__(self):

        # fetch user cmd selections
        self.parser = argparse.ArgumentParser()
        self.parse_agrs()

        n_gpu = self.parser.parse_args().n_gpu
        self.device = "cuda:3"

    def parse_agrs(self) -> None:
        """ Parse all arguments selected in execution from the user
        """
        parser = argparse.ArgumentParser()

        # Data loader settings
        self.parser.add_argument("--dataset_name", type=str, default="imageclef", choices=["iu_xray", "imageclef"], help="the dataset to be used.")

        #self.parser.add_argument("--dataset_images_path", type=str, default="/media/SSD_2TB/imageclef2024/full_dataset/original_images")
        self.parser.add_argument("--dataset_images_path", type=str, default="/media/SSD_2TB/imageclef2024/test")
        
        self.parser.add_argument("--dataset_captions_path_train", type=str, default="/media/SSD_2TB/imageclef2024/original_dataset/unprocessed_captions/image_caps_original/train_captions_unprocessed.csv")
        self.parser.add_argument("--dataset_captions_path_valid", type=str, default="/media/SSD_2TB/imageclef2024/original_dataset/unprocessed_captions/image_caps_original/valid_captions_unprocessed.csv")
        #self.parser.add_argument("--dataset_captions_path_test", type=str, default="/media/SSD_2TB/imageclef2024/original_dataset/unprocessed_captions/image_caps_original/train_captions_unprocessed.csv")
        self.parser.add_argument("--dataset_captions_path_test", type=str, default="/home/pkaliosis/instructBLIP/imageclef2024_test.csv")

        self.parser.add_argument("--dataset_concepts_path_test", type=str, default="/home/pkaliosis/dmm_stats_24/test_submission_file_foivos2024_k33_MFtag_upto4_th_058_065_no_modal.csv")
        self.parser.add_argument("--dataset_concepts_mapper", type=str, default="/home/pkaliosis/cnn_rnn/ImageCLEFmedical_Caption_2023_cui_mapping.csv")
        
        # Captions settings
        self.parser.add_argument("--max_length", type=int, default=80, help="the maximum sequence length of the reports.")
        self.parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')

        self.parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
        self.parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

        args = self.parser.parse_args()
        return args
    
    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:7' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def load_imageclef_data(self) -> dict:
        """ Loads ImageCLEF dataset from directory

        Returns:
            tuple[dict, dict]: Image vectors, captions in dictionary format, with keys to be the Image IDs.
        """
        # Parse the subsets' paths
        dataset_captions_path_train = self.parser.parse_args().dataset_captions_path_train
        dataset_captions_path_valid = self.parser.parse_args().dataset_captions_path_valid
        dataset_captions_path_test = self.parser.parse_args().dataset_captions_path_test

        # Load the three subsets into pandas dataframes
        clef_captions_df_train = pd.read_csv(dataset_captions_path_train, names=["ID", "caption"], header=0)
        clef_captions_df_valid = pd.read_csv(dataset_captions_path_valid, names=["ID", "caption"], header=0)
        clef_captions_df_test = pd.read_csv(dataset_captions_path_test, names=["ID", "caption"], header=0)

        # print the training subset's head!
        print(clef_captions_df_train.head())

        #clef_captions_df_train = clef_captions_df_train.head(20000)
        #clef_captions_df_valid = clef_captions_df_valid.head(3500)
        #clef_captions_df_test = clef_captions_df_test.iloc[2000:2020]

        # and now zip them into a dict!
        captions_train = dict( zip( clef_captions_df_train.ID.to_list(), clef_captions_df_train.caption.to_list() ) )
        captions_valid = dict( zip( clef_captions_df_valid.ID.to_list(), clef_captions_df_valid.caption.to_list() ) )
        captions_test = dict( zip( clef_captions_df_test.ID.to_list(), clef_captions_df_test.caption.to_list() ) )

        concept_mapper_path = self.parser.parse_args().dataset_concepts_mapper

        concepts_mapper = pd.read_csv(concept_mapper_path, sep="\t", header=None, names=['cui', 'concept'])

        # Build a mapper
        self._concepts_dict = {}
        for row in concepts_mapper['concept']:
            mapper = concepts_mapper.loc[concepts_mapper['concept'] == row].values.flatten().tolist()
            self._concepts_dict[mapper[0]] = mapper[1]
        
        return captions_train, captions_valid, captions_test

    
    def load_tags_data(self) -> dict:
        """ Loads ImageCLEF dataset from directory

        Returns:
            tuple[dict, dict]: Image vectors, captions in dictionary format, with keys to be the Image IDs.
        """
        # get dataset path
        dataset_concepts_path_test = self.parser.parse_args().dataset_concepts_path_test

        clef_concepts_df_test = pd.read_csv(dataset_concepts_path_test, sep=',', header=None, names=['ID', 'cuis'])

        print("CONCEPTS TEST:", clef_concepts_df_test.head())

        concepts_test = dict( zip( clef_concepts_df_test.ID.to_list(), clef_concepts_df_test.cuis.to_list() ) )
        
        return concepts_test



    def split_data(self, captions_train:dict, captions_valid:dict, captions_test:dict):

        train_ids = list(captions_train.keys())
        dev_ids = list(captions_valid.keys())
        test_ids = list(captions_test.keys())

        return train_ids, dev_ids, test_ids


    def split_(self, dict_to_split_train:dict, dict_to_split_val:dict, dict_to_split_test:dict):

        train, dev, test = list(), list(), list()
        for k in dict_to_split_train.keys():
            train.append((dict_to_split_train[k].split(';')))
        
        for k in dict_to_split_val.keys():
            dev.append((dict_to_split_val[k].split(';')))

        print(dict_to_split_test)
        for k in dict_to_split_test.keys():
            test.append((dict_to_split_test[k].split(';')))
                
        return train, dev, test

    
    # Creating the training function. This will be called in the main function. It is run depending on the epoch value.
    # The model is put into train mode and then we wnumerate over the training loader and passed to the defined network
    def train(self, epoch, instruction_, model, processor, device, loader, optimizer):
        self.model.train() # switch into training mode
        running_loss = 0 # define loss
        batch_counter = 0 # define batch counter

        # and start the training loop!
        for _, data in tqdm(enumerate(loader, 0)):
            image = data[0]
            caption = data[1]
            ids = data[2]

            batch_counter += 1

            instruction = [instruction_ for i in range(len(caption))]

            inputs = processor(images=image, text=instruction, return_tensors="pt")
            inputs = inputs.to(device)

            labels = processor.tokenizer(caption, padding="max_length", max_length=40, truncation=True, return_tensors="pt")
            labels["input_ids"] = torch.tensor([[-100 if x == processor.tokenizer.pad_token_id else x for x in labels["input_ids"][i].tolist()] for i in range(len(caption))])
            labels = torch.tensor(labels.input_ids).to(device)

            outputs = model(**inputs, labels = labels)

            loss = outputs.loss
            running_loss += loss.item()

            if _%500==0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        del inputs
        del labels
            

        epoch_loss = running_loss / batch_counter
        print(f'Epoch: {epoch}, Loss:  {epoch_loss}, Batch Counter: {batch_counter}')


    def validate(self, epoch, instruction_, model, processor, device, loader):
        self.model.eval()
        predictions = []
        actuals = []
        val_batch_counter = 0
        running_val_loss = 0

        with torch.no_grad():
            for _, data in tqdm(enumerate(loader, 0)):
                val_batch_counter += 1

                image = data[0]
                caption = data[1]
                ids = data[2]

                instruction = [instruction_ for i in range(len(caption))]

                inputs = processor(images=image, text=instruction, return_tensors="pt")
                inputs = inputs.to(device)

                labels = processor.tokenizer(caption, padding="max_length", max_length=40, truncation=True, return_tensors="pt")
                labels["input_ids"] = torch.tensor([[-100 if x == processor.tokenizer.pad_token_id else x for x in labels["input_ids"][i].tolist()] for i in range(len(caption))])
                labels = torch.tensor(labels.input_ids).to(device)

                val_outputs = model(**inputs, labels = labels)

                val_loss = val_outputs.loss
                running_val_loss += val_loss.item()
            
            del inputs
            del labels
                

            epoch_val_loss = running_val_loss / val_batch_counter
            print(f'Epoch: {epoch}, Validation Loss:  {epoch_val_loss}, Batch Counter: {val_batch_counter}')
            if (epoch_val_loss < self.best_loss):
                    self.best_loss = epoch_val_loss
                    self.early_stopping_counter = 0

                    # save model in order to retrieve at the end...
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
            else:
                    self.early_stopping_counter += 1

    def is_subset(self, lst1, lst2):
        return set(lst1).issubset(set(lst2))

    def remove_subsets(self, list_of_lists):
        result = []
        for i, inner_list in enumerate(list_of_lists):
            is_subset_of_any = False
            for j, other_list in enumerate(list_of_lists):
                if i != j and self.is_subset(inner_list, other_list):
                    is_subset_of_any = True
                    break
            if not is_subset_of_any:
                result.append(inner_list)
        return result

    
    def test(self, epoch, instruction_, model, processor, device, loader, hist_file_path, mmc_sim_file_path, tokenizer):
        model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for _, data in tqdm(enumerate(loader, 0)):

                image = data[0]
                caption = data[1]
                tags = data[2]
                """tt = tags
                print(type(tt) is tuple)
                if type(tt) is tuple:
                    tags = tags[0].split(';')
                    for i, t in enumerate(tags):
                        tags[i] = self._concepts_dict[t]"""


                instruction = [instruction_ for i in range(len(caption))]

                #dmm = DMM(hist_train_file=hist_file_path, mmc_sim_file=mmc_sim_file_path)

                # instantiating a list of LogitsProcessor instances
                # using our custom ABCLogits class
                #alpha = 0.2
                #logits_processor = LogitsProcessorList([DMMLogits(dmm, tags, alpha, tokenizer)])
                inputs = processor(images=image, text=instruction, return_tensors="pt").to(device)


                outputs = model.generate(
                        **inputs,
                        do_sample=False,
                        num_beams=5,
                        max_length=120,
                        min_length=5,
                        #logits_processor=logits_processor
                    )

                generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
                #print("generated text:", generated_text)

                if _%100==0:
                    print(f'Completed {_}')
                    
                predictions.extend(generated_text)
                actuals.extend(caption)
        
        return predictions, actuals


    def main(self):
        self.TRAIN_BATCH_SIZE = 4   # input batch size for training (default: 64)
        self.VALID_BATCH_SIZE = 4  # input batch size for testing (default: 1000)
        self.TEST_BATCH_SIZE = 1
        TRAIN_EPOCHS = 40       # number of epochs to train (default: 10)
        VAL_EPOCHS = 1
        TEST_EPOCHS = 1 
        LEARNING_RATE = 5e-6    # learning rate (default: 0.01)
        SEED = 42               # random seed (default: 42)
        self.epoch_ = -1

        # Set random seeds and deterministic pytorch for reproducibility
        torch.manual_seed(SEED) # pytorch random seed
        np.random.seed(SEED) # numpy random seed
        #torch.backends.cudnn.deterministic = True

        # load data in pandas dataframe form
        captions_train, captions_valid, captions_test = self.load_imageclef_data()

        # load image ids individually
        #print("captions train:", captions_train)
        #print("captions test ids:", captions_test)
        train_ids, dev_ids, test_ids = self.split_data(captions_train, captions_valid, captions_test)

        tags_test = self.load_tags_data()

        # load the captions individually
        train_labels, dev_labels, test_labels = self.split_(captions_train, captions_valid, captions_test)

        # define images path
        self.images_path = self.parser.parse_args().dataset_images_path

        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")

        # define the instruction to be followed during training
        #instruction = "Describe the radiology image."
        instruction = 'You are an experienced radiologist. You are being given radiology images along with a short medical diagnosis. Generate a descriptive caption that highlights the location, nature and severity of the abnormality of the radiology image.'

        self.model = model.to(self.device)
        self.processor = processor
        
        for i, param in enumerate(self.model.vision_model.encoder.layers.parameters()):
            param.requires_grad = False

        for i, param in enumerate(self.model.language_model.encoder.parameters()):
            param.requires_grad = False
        
        c = 0
        for i, param in enumerate(self.model.language_model.decoder.parameters()):
            if i <= 334:
                param.requires_grad = False
            c += 1

        c2 = 0
        for i, param in enumerate(self.model.qformer.encoder.layer.parameters()):
            c2+=1
            if i <= 190:
                param.requires_grad = False

        true_, false_ = 0, 0
        for i, param in enumerate(self.model.parameters()):   
            g = param.requires_grad
            if (g):
                true_ += 1
            else:
                false_ += 1
            
        print('Require grad:', true_)
        print('Frozen:', false_)

        _ = list()

        self.train_dataset = CustomVisionDataset(captions_train, train_ids, _, processor, 'train')
        self.val_dataset = CustomVisionDataset(captions_valid, dev_ids, _, processor, 'validation')
        self.test_dataset = CustomVisionDataset(captions_test, test_ids, tags_test, processor, 'test')

        self.hist_file_path = '/home/pkaliosis/dmm_stats_24/hist_train.pkl'
        self.mmc_sim_file_path = '/home/pkaliosis/dmm_stats_24/median_max_cos_c.pkl'

        # Defining the parameters for creation of dataloaders
        train_params = {
            'batch_size': self.TRAIN_BATCH_SIZE,
            'shuffle': True,
            'num_workers': 4
        }

        val_params = {
            'batch_size': self.VALID_BATCH_SIZE,
            'shuffle': False,
            'num_workers': 4
        }

        test_params = {
            'batch_size': self.TEST_BATCH_SIZE,
            'shuffle': False,
            'num_workers': 4
        }


        # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
        training_loader = DataLoader(self.train_dataset, **train_params)
        val_loader = DataLoader(self.val_dataset, **val_params)
        test_loader = DataLoader(self.test_dataset, **test_params)

        print('Running InstructBLIP-Flan-T5xl on:', self.device)

        # Defining the optimizer that will be used to tune the weights of the network in the training session. 
        optimizer = torch.optim.Adam(params =  self.model.parameters(), lr=LEARNING_RATE)

        print('Initiating intruction-based fine-tuning for the model on our dataset')

        self.best_loss = 1000000
        self.early_stopping_counter = 0
        for epoch in tqdm(range(TRAIN_EPOCHS)):
            self.train(epoch, instruction, self.model, self.processor, self.device, training_loader, optimizer)
            self.validate(epoch, instruction, self.model, self.processor, self.device, val_loader)

            # check for early stopping
            print('Early stopping counter:', self.early_stopping_counter)
            if ((self.early_stopping_counter >= 3) or (epoch == (TRAIN_EPOCHS - 1))):

                del self.model
                del model
                del optimizer

                #model_path = BEST_MODEL_PATH
                model_path = "/home/pkaliosis/instructBLIP/saved_models/2024_instructblip_1e-5.pt"
                model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
                state_dict = torch.load(model_path, map_location = 'cpu')
                model.load_state_dict(state_dict)
                best_model = model.to(self.device)
                break

        print('Now generating summaries on our fine tuned model for the test dataset and saving it in a dataframe')
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
        for epoch in range(TEST_EPOCHS):
            #predictions, actuals = self.test(epoch, self.tokenizer, best_model, self.processor, self.device, test_loader)
            predictions, actuals = self.test(epoch, instruction, best_model, self.processor, self.device, test_loader, self.hist_file_path, self.mmc_sim_file_path, tokenizer)

            print('predictions:', predictions)
            print('actuals:', actuals)

            with open(RESULTS_PATH, 'w') as out_test:
                for i, pred in enumerate(predictions):
                    out_test.write(test_ids[i] + '|' + pred + '\n')
            print('Results saved!')

            final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
            #final_df.to_csv(GENERATIONS_PATH)
            print('Output Files generated for review')


if __name__ == '__main__':
    # define tokenizer

    instruct_blip = InstructBLIP()

    instruct_blip.main()
