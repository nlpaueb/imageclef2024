# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
import torch
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import AutoImageProcessor
from PIL import Image
import os



class CustomVisionDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, ids, tags_test, image_processor, mode:str):
        
        self.mode = mode
        self.data = dataframe
        self.tags_data = tags_test
        self.ids = ids
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.dict_keys = list(self.data.keys())
        normalize = Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std)

        # -------------------------------------------------------------------------------
        # image transformations
        if mode == 'train':
           self._transforms = transforms.Compose([
                                    transforms.RandomRotation(30),
                                    #transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
                            

        
        elif mode == 'validation':
            self._transforms = transforms.Compose([
                                    transforms.RandomRotation(30),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

        else:
           self._transforms = transforms.Compose([
                                    #transforms.RandomRotation(30),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

        self.dataset_images_path = "/media/SSD_2TB/imageclef2024/test"

    def __len__(self):
        return len(self.data)

    

    def __getitem__(self, index):
        
        data_dir = self.dataset_images_path

        caption = self.data[self.dict_keys[index]]

        if self.mode == 'test':
            tags = self.tags_data[self.dict_keys[index]]

            #print('tags:', tags)

        image = Image.open(os.path.join(data_dir, self.ids[index] + '.jpg'))
        image = image.convert('RGB')
        image = self._transforms(image)

        #image = self.image_processor(image, return_tensors="pt").pixel_values
        #print('img:', image)
        #image = torch.tensor(image)
        #print('image type:', type(image))
        
        
        if self.mode == 'test':
            return image, caption, tags, self.ids[index]
        else:
            return image, caption, self.ids[index]