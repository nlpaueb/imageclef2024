import pandas as pd
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from tqdm import tqdm

# Load model and tokenizer
model_path = '/home/msamprovalaki/models/instructblip-flan-t5-large-patience-7-valid/'
# Alternative model paths
# model_path = '/home/msamprovalaki/models/instructblip-flan-t5-large-patience-7-valid-diff-prompt/'
# model_path = '/home/msamprovalaki/models/instructblip-flan-t5-large-valid-set-1neighbor/'

finetuned_model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load data
test_neighbors_file = '/media/SSD_2TB/imageclef2024/test_neighbors/test_images_w_neighbor_captions5.csv'
blip_predictions_file = '/media/SSD_2TB/imageclef2024/captioning_submissions/instructBLIP_only.csv'

test_neighbors = pd.read_csv(test_neighbors_file)
blip_predictions = pd.read_csv(blip_predictions_file, header=None, sep='|')
blip_predictions.columns = ['ID', 'Generated_Caption']
blip_predictions['ID'] = blip_predictions['ID'].apply(lambda x: x + '.jpg')

# Merge neighbors and BLIP predictions
df = pd.merge(test_neighbors, blip_predictions, on='ID')

# Prepare model for GPU
device = torch.device('cuda:4')
model.to(device)
model.eval()

# Define batch processing parameters
batch_size = 8
num_batches = len(df) // batch_size + 1

# Initialize list to store final answers
answers = []

# Process data in batches
for batch_idx in tqdm(range(num_batches), desc="Processing Batches"):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(df))
    batch_df = df.iloc[start_idx:end_idx]

    # Generate prompts
    prompts = [
        f'You are a medical professional tasked with enhancing the generated caption "{row["Generated_Caption"]}" by incorporating insights from neighboring captions "{row["Captions"]}". Craft a comprehensive caption that precisely depicts the image\'s content.'
        for _, row in batch_df.iterrows()
    ]

    inputs = tokenizer(prompts, return_tensors='pt', max_length=300, padding='max_length', truncation=True)
    input_ids = inputs['input_ids'].to(device)
    
    outputs = model.generate(input_ids, max_length=30, num_beams=5, early_stopping=True)
    batch_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    answers.extend(batch_answers)

# Add final captions to the dataframe
df['Final_Caption'] = answers

# Prepare final predictions dataframe
dev_predictions_prompt = df[['ID', 'Final_Caption']]
dev_predictions_prompt = dev_predictions_prompt.rename(columns={"Final_Caption": "Generated_Caption"})
dev_predictions_prompt['ID'] = dev_predictions_prompt['ID'].apply(lambda x: x.replace('.jpg', ''))

# Save final predictions to a CSV file
output_file = '/media/SSD_2TB/imageclef2024/captioning_submissions/commas/temp.csv'
dev_predictions_prompt.to_csv(output_file, sep='|', index=False, header=None)

print("Test data saved.")
