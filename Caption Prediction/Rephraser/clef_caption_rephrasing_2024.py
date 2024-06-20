import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import clinicalT5_generator as llm
import caption_rephrase_loader_2024 as dataset
from tqdm import trange
from common import *
import re

# Define the instance variables
max_epochs = 20
train_batch_size = 5
val_batch_size = 1
dev_batch_size = 1

# Define the data containers
train_data = dataset.RephraseTextLoader(split='train_dmm')
dev_data = dataset.RephraseTextLoader(split='dev_dmm')

# Compute the maximum input/output lengths
max_input_len = dataset.max_length(train_data.impression)
max_output_len = dataset.max_length(train_data.findings)

# Check hardware accelerator availability
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Define the model architecture and optimizer
model = llm.ClinicalT5CLEF(max_input_len=max_input_len, max_output_len=max_output_len, num_beams=5,
                           device=device).to(device)

optimizer = torch.optim.Adam(model.parameters())

# Define the data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=dev_batch_size, shuffle=False)

# Define the training script
max_patience = 20
patience = max_patience
train_losses = []
dev_scores = []
dev_losses = []
# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPOCHS = trange(max_epochs, desc='Epoch: ', leave=True)
for epoch in EPOCHS:  # loop over the dataset multiple timesΣ
    # Model training
    model.train()
    train_loss = 0.0
    for i, (input, target) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        batch_loss = model(list(input), list(target)).loss
        train_loss += batch_loss.detach().cpu().numpy().item()
        batch_loss.backward()
        optimizer.step()
    train_losses.append(train_loss/len(train_data))

    save_dir = "/media/georgiosm/HDD_8TB/imageclef2024/weights/rephraser"
    torch.save(model.state_dict(), os.path.join(save_dir, "DMM_ClinicalT5_epoch" + str(epoch) + ".pth"))

    # Model testing
    dev_loss = 0.0
    DEV_SCORE = []
    for i, (input, target) in enumerate(dev_loader, 0):
        prediction = model.generate(list(input))
        bert_score = model.evaluate(prediction[0], target)
        DEV_SCORE.append(bert_score)
        torch.cuda.empty_cache()
    dev_scores.append(np.average(DEV_SCORE))

    BERTScore_plot(dev_scores, os.path.join(save_dir, "DMM_ClinicalT5_BertScores.jpg"))

print("Best epoch:" + str(np.argmax(np.array(dev_scores))))
print("Best score:" + str(np.max(np.array(dev_scores))))

print("Training completed.")