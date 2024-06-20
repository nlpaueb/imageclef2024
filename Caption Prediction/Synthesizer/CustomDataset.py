import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image_id = item['ID']
        caption = item['Captions']
        concatenated_captions = [' '.join(caption)]
        generated_caption = item['Generated_Caption']
        target_text = item['Gold_Caption']
        neighbors = concatenated_captions

        prompt = f'You are a medical professional tasked with enhancing the generated caption "{generated_caption}" by incorporating insights from neighboring captions "{neighbors}". Craft a comprehensive caption that precisely depicts the image\'s content.'

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        target_ids = self.tokenizer.encode(target_text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        
        return input_ids, target_ids