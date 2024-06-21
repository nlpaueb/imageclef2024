
# AUEB NLP Group at ImageCLEFmedical Caption 2024

### Details
This repository includes the approaches that the AUEB NLP Group experimented with during its participation in the 8th edition of the [ImageCLEFmedical Caption evaluation campaign](https://www.imageclef.org/2024/medical/caption), including both Concept Detection and Caption Prediction tasks.

### Concept Detection
Initially, we extensively explored a CNN+FFNN framework, experimenting with various image encoders. Additionally, we used a neural image retrieval approach by integrating a 𝑘-nearest neighbors (𝑘-NN) algorithm, which selects 𝑘 neighbors and aggregates tags based on their frequency among the neighbors. Furthermore, we submitted several ensembles of the aforementioned systems. The ensembles employed strategies such as union-based and intersection-based aggregation.

### Caption Predictions
Our submissions for the Caption Prediction sub-task focused on four primary systems. The first system employs an InstructBLIP model, while the remaining submissions build on this model using techniques such as rephrasing and synthesizing. Finally, we implemented an innovative guided-decoding mechanism, [DMMCS](https://github.com/nlpaueb/dmmcs), which leverages information from the tags predicted by our CNN+𝑘-NN classifier in the Concept Detection task to improve the generated caption

The paper for this project will be announced soon.

### Cite
If you find our work useful please cite our paper:


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


