import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import mimic_cxr_loader as dataset
from tqdm import trange
from common import *
import string
import re

class ClinicalT5Model(nn.Module):
    def __init__(self, max_input_len, max_output_len, num_beams, root=None, device='cuda:0'):
        super().__init__()
        self.root = root if root is not None \
            else "/media/georgiosm/HDD_8TB/clinical-t5/1.0.0/Clinical-T5-Large"
        self.tokenizer = AutoTokenizer.from_pretrained(self.root)
        self.seq2seq = AutoModelForSeq2SeqLM.from_pretrained(self.root)
        self.metrics = evaluate.load("bertscore")
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.num_beams = num_beams
        self.device = device

    def forward(self, input_text, source_text):
        tokenized_input = self.tokenize(input_text)
        tokenized_source = self.tokenize(source_text)

        output = self.seq2seq(
            input_ids=tokenized_input.input_ids.to(self.device, non_blocking=True),
            attention_mask=tokenized_input.attention_mask.to(self.device, non_blocking=True),
            labels=tokenized_source.input_ids.to(self.device, non_blocking=True),)

        return output

    def generate(self, input_text):
        tokenized_input = self.tokenize(input_text)

        output = self.seq2seq.generate(
            input_ids=tokenized_input.input_ids.to(self.device, non_blocking=True),
            attention_mask=tokenized_input.attention_mask.to(self.device, non_blocking=True),
            max_length=self.max_output_len,  # type: ignore
            num_beams=self.num_beams,
            num_return_sequences=1,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

        output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        return output_text, torch.stack(list(output.scores), dim=0)

    def tokenize(self, input_text):
        return self.tokenizer(
            input_text,
            padding="longest",
            max_length=self.max_input_len,  # type: ignore
            truncation=True,
            return_tensors="pt",
        )

    """
    Evaluate predictions using scores from DistilBERT.
    """
    def evaluate(self, refs, hyps):
        self.metrics.add_batch(predictions=refs, references=hyps)
        results = self.metrics.compute(model_type='distilbert-base-uncased', num_layers=5, all_layers=False, idf=False,
                                       lang='en', rescale_with_baseline=True, baseline_path=None)
        return np.mean(results["f1"])


class ClinicalT5CLEF(ClinicalT5Model):
    def __init__(self, max_input_len, max_output_len, num_beams, root=None, device='cuda:0'):
        super().__init__(max_input_len, max_output_len, num_beams, root, device)
        self.metrics = evaluate.load("bertscore", map_location=device)

    """
    Evaluate predictions using scores from DistilBERT.
    """
    def evaluate(self, refs, hyps):
        # Remove punctuation from string
        translator = str.maketrans('', '', string.punctuation)

        # Regex for numbers
        number_regex = re.compile(r'\d+')

        bert_scores = []
        counter = 0

        for i in range(len(refs)):
            # Get candidate and GT caption
            candidate_caption = refs[i]
            gt_caption = hyps[i]

            # Optional - Go to lowercase
            # if not type(self).case_sensitive:
            if True:
                candidate_caption = candidate_caption.lower()
                gt_caption = gt_caption.lower()

            # replace numbers with the token 'number'
            candidate_caption = number_regex.sub('number', candidate_caption)
            gt_caption = number_regex.sub('number', gt_caption)

            # Remove punctuation using the translator
            candidate_caption = candidate_caption.translate(translator)
            gt_caption = gt_caption.translate(translator)

            # Calculate BERTScore for the current caption
            try:
                # If both the GT and candidate are empty, assign a score of 1 for this caption
                if len(gt_caption) == 0 and len(candidate_caption) == 0:
                    bert_score["f1"] = 1
                else:
                    # Calculate the BERTScore
                    bert_score = self.metrics.compute(predictions=[candidate_caption], references=[gt_caption],
                                                      model_type='microsoft/deberta-xlarge-mnli')
                bert_scores.append(bert_score["f1"])

            # Handle problematic cases where BERTScore calculation is impossible
            except Exception as e:
                print(e)
                # raise Exception('Problem with {} {}', gt_caption, candidate_caption)

        return np.mean(bert_scores)

