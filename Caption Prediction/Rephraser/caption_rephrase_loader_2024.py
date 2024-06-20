import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from common import *

class RephraseTextLoader(Dataset):
    def __init__(self, root=None, split='trainGM'):
        # Initialize variables
        self.root = root if root is not None \
            else "/media/georgiosm/HDD_8TB/imageclef2024/generated_captions"
        self.findings = []
        self.impression = []

        # Process the patients information
        findings = read_file(self.root, split.split("_")[0] + "_gt.csv")
        impression = read_file(self.root, split + "_pred.csv")
        for i in tqdm(range(len(findings))):
            try:
                self.findings.append(findings[i].split("|")[1])
                self.impression.append(impression[i].split("|")[1])
            except:
                pass

    def __getitem__(self, index):
        in_caption = self.findings[index]
        out_caption = self.impression[index]

        return in_caption, out_caption

    def __len__(self):
        return len(self.findings)


class RephraseTextTestLoader(Dataset):
    def __init__(self, root=None, split='test'):
        # Initialize variables
        self.root = root if root is not None \
            else "/media/georgiosm/HDD_8TB/imageclef2024/generated_captions_test"
        self.findings = []
        self.imgIDs = []

        # Process the patients information
        findings = read_file(self.root, split + "_pred.csv")
        for i in tqdm(range(len(findings)-1)):
            self.findings.append(findings[i].split("|")[1])
            self.imgIDs.append(findings[i].split("|")[0])

    def __getitem__(self, index):
        in_caption = self.findings[index]
        imgID = self.imgIDs[index]

        return in_caption, imgID

    def __len__(self):
        return len(self.findings)