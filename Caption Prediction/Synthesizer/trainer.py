import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer from Hugging Face
def loader():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    return tokenizer, model

# this function is used to calculate the custom loss between logits and gold captions in order to train the model
def custom_loss(logits, gold_captions):
    logits_flat = logits.view(-1, logits.shape[-1])
    gold_captions_flat = gold_captions.view(-1)
    loss = F.cross_entropy(logits_flat, gold_captions_flat)
    return loss

# Load data and neighbors
def data_and_neighbors():
    
    # Training dataset that contains the captions and the neighbors for each training image
    instructblip = pd.read_csv('/home/msamprovalaki/datasets/instructblip/train_neighbors.csv')
    instructblip = instructblip.drop(columns=['Gold_Caption']) # removing the gold captions from the training images

    instructblip_valid = pd.read_csv('/home/msamprovalaki/datasets/instructblip/valid_neighbors_with_prediction.csv')
    instructblip_valid = instructblip_valid.drop(columns=['Gold_Caption']) # removing the gold captions from the validation images

    # Predictions for the training dataset using InstructBLIP, separated by |
    predictions = pd.read_csv('/home/msamprovalaki/instructBLIP/results/train_data_mysplit_generated_more_layers.csv', sep='|', header=None)
    predictions.columns = ['ID', 'Generated_Caption']
    instructblip = instructblip.merge(predictions, on='ID')

    # Predictions for the validation dataset using InstructBLIP, separated by |
    predictions_valid = pd.read_csv('/home/msamprovalaki/instructBLIP/results/valid_data_mysplit_generated_more_layers.csv', sep='|', header=None)
    predictions_valid.columns = ['ID', 'Generated_Caption']
    instructblip_valid = instructblip_valid.merge(predictions_valid, on='ID')
     
    # Split the data into training and validation sets
    train_data = instructblip
    valid_data = instructblip_valid

    return train_data, valid_data

def train_model(train_loader, valid_loader, model, optimizer, num_epochs=30, patience=7):
    model.to(device)
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            input_ids, target_ids = batch
            # squeeze the tensor to remove the extra dimension
            input_ids = input_ids.squeeze(1).to(device)
            target_ids = target_ids.squeeze(1).to(device)
            # clear the gradients
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=target_ids)
            logits = outputs.logits
            # calculate the loss
            loss = custom_loss(logits, target_ids)
            # backpropagate the loss
            loss.backward()
            # update the weights
            optimizer.step()
            total_loss += loss.item()

        # calculate the average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Avg Loss: {avg_loss:.4f}")
        
        val_loss = 0
        model.eval()
        # torch.no_grad() disables the gradient calculation
        with torch.no_grad():
            for input_ids, target_ids in tqdm(valid_loader):
                # squeeze the tensor to remove the extra dimension
                input_ids = input_ids.squeeze(1).to(device)
                target_ids = target_ids.squeeze(1).to(device)
                outputs = model(input_ids=input_ids, labels=target_ids)
                logits = outputs.logits
                loss = custom_loss(logits, target_ids)
                val_loss += loss.item()

        val_loss /= len(valid_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Valid Avg Loss: {val_loss:.4f}")
        print(50*'=')

        # Early stopping if the validation loss does not improve for selected number of epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            # increment the counter if the validation loss does not improve
            counter += 1
            if counter >= patience:
                print("Validation loss has not improved for {} epochs. Early stopping...".format(patience))
                break

    print("Training complete!")

def save_model_and_tokenizer(model, tokenizer, path):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print("Model saved!")

if __name__ == "__main__":
    # Load model and tokenizer
    tokenizer, model = loader()
    print("Model and tokenizer loaded!")
    # Load data and neighbors
    train_data, valid_data = data_and_neighbors()
    print("Data loaded!")
    # Create custom dataset and dataloaders
    train_dataset = CustomDataset(train_data, tokenizer, max_length=300)
    valid_dataset = CustomDataset(valid_data, tokenizer, max_length=300)
    print("Datasets created!")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)
    print ("Dataloaders created!")
    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)
    print("Training started...")
    # Train the model
    train_model(train_loader, valid_loader, model, optimizer)
    print("Training completed...")
    # Save the model and tokenizer
    save_model_and_tokenizer(model, tokenizer, '/home/msamprovalaki/models/instructblip-flan-t5-large-patience-7')
    print("Model saved!")
