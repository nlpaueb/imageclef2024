
# AUEB NLP Group at ImageCLEFmedical Caption 2024

## Details
This repository includes the approaches that the AUEB NLP Group experimented with during its participation in the 8th edition of the [ImageCLEFmedical Caption evaluation campaign](https://www.imageclef.org/2024/medical/caption), including both Concept Detection and Caption Prediction tasks.

### Concept Detection
Initially, we extensively explored a CNN+FFNN framework, experimenting with various image encoders. Additionally, we used a neural image retrieval approach by integrating a 𝑘-nearest neighbors (𝑘-NN) algorithm, which selects 𝑘 neighbors and aggregates tags based on their frequency among the neighbors. Furthermore, we submitted several ensembles of the aforementioned systems. The ensembles employed strategies such as union-based and intersection-based aggregation.

### Caption Predictions
Our submissions for the Caption Prediction sub-task focused on four primary systems. The first system employs an InstructBLIP model, while the remaining submissions build on this model using techniques such as rephrasing and synthesizing. Finally, we implemented an innovative guided-decoding mechanism, [DMMCS](https://github.com/nlpaueb/dmmcs), which leverages information from the tags predicted by our CNN+𝑘-NN classifier in the Concept Detection task to improve the generated caption

The paper for this project will be announced soon.

## Installation
To get started with our models, follow these steps to clone the repository and install the required packages. We recommend using a virtual environment for package installation to ensure a clean and isolated setup.

### Step 1: Clone the repository

```
git clone git@github.com:nlpaueb/imageclef2024.git
cd imageclef2024
```

### Step 2: Create and activate a virtual environment

We have tested our framework for Conda environment.

#### Conda

```
conda create -n imageclef_env python=3.9
conda activate imageclef_env
pip install -r requirements.txt
```
## Usage

### Step 1: Generate the initial captions using any image captioning model, we have used the InstructBLIP model. (for both Synthesizer and Rephraser)

You can train and/or use an InstructBLIP model, or any image captioning model of your choice, to generate the initial captions. These captions will then be refined using our models.

```
python3 instructBLIP-ft.py --config ../config/config.json
```

### Step 2: Create your neighbor dataset (only for the Synthesizer)
You need to create a CSV file with two columns. The first column should contain the path for each image, and the second column should indicate the number of neighbors you have identified. Currently, we have conducted experiments using 1, 3, or 5 neighbors, with the best results obtained from using 5 neighbors.


### Step 3: Run training and/or inference 
You can use both of our models together or each one separately.

### Rephraser

### Synthesizer
#### Train

```
python3 trainer.py 
```

#### Inference

```
python3 synthesizer-inference.py 
```


Please make sure to adjust the paths in the file to your own local paths and directories.


### Citation
If you would like to use our work, please cite us using the following bibtex reference:


```
@inproceedings{samprovalaki-2024-aueb,
 address = {Grenoble, France},
 author = {Marina Samprovalaki and Anna Chatzipapadopoulou and Georgios Moschovis and Foivos Charalampakos and Panagiotis Kaliosis and John Pavlopoulos and Ion Androutsopoulos},
 booktitle = {CLEF2024 Working Notes},
 publisher = {CEUR-WS.org},
 series = {CEUR Workshop Proceedings},
 title = {{AUEB NLP Group at ImageCLEFmedical Caption 2024}},
 year = {2024}
}
```
