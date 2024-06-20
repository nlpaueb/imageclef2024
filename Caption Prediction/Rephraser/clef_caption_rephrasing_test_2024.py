import torch
import clinicalT5_generator as llm
import caption_rephrase_loader_2024 as dataset
from helpers import write
from tqdm import tqdm

predictions = {}

test_batch_size = 1

# Define the data containers
test_data = dataset.RephraseTextTestLoader(split='test_dmm')

# Compute the maximum input/output lengths
max_input_len = 128
max_output_len = 5630

# Check hardware accelerator availability
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Define the model architecture and optimizer
model = llm.ClinicalT5CLEF(max_input_len=max_input_len, max_output_len=max_output_len, num_beams=5,
                           device=device).to(device)

model.load_state_dict(
    torch.load("/media/georgiosm/HDD_8TB/imageclef2024/weights/submissions2024/DMM_ClinicalT5_epoch0.pth"))

# Define the data loaders
test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

# Model testing
for i, (input, imgID) in tqdm(enumerate(test_loader, 0), total=len(test_data)):
    prediction = model.generate(list(input))
    predictions[imgID[0]] = prediction[0][0]
    torch.cuda.empty_cache()

output = ""
for imgID in predictions.keys():
    output += imgID + "," + predictions[imgID] + "\n"

write(output, "/media/georgiosm/HDD_8TB/imageclef2024/generated_captions_submit", "test_dmm_submission.csv")
