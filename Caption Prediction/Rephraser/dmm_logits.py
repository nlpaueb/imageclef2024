from dmm import DMM
from transformers import LogitsProcessor
import torch

class DMMLogits(LogitsProcessor):
  def __init__(self, dmm, tags, alpha, tokenizer):
    """
    vocab is a dictionary where the keys are tokens
    and the values are the corresponding ids.
    """
    self.dmm = dmm
    self.tags = tags
    self.alpha = alpha
    self.tokenizer = tokenizer

  def normalize_dmm(self, value):
    xmin = 0
    xmax = 1.5

    norm = ((value - xmin) / (xmax-xmin))

    return norm

  def normalize_lm(self, tensor_):
    xmin = 2
    xmax = 22

    norm = ((tensor_ - xmin) / (xmax-xmin))

    return norm

  def __call__(self, input_ids, scores):
    # for every beam (partially generated sentence)
    hist_conf = list()
    final_scores = list()
    #print('scores shape:', scores.shape)
    for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
        
        running_caption = self.tokenizer.decode(beam_input_ids)
        #hist_conf = self.dmm.dmm_handler(running_caption, self.tags, calc_dmm_loss=False)
        hist_conf.append(self.dmm.dmm_handler(running_caption, self.tags, calc_dmm_loss=False))
        #print('hist conf:', hist_conf)
    
    final_scores = list()
    
    for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):

        #print('beam scores:', beam_scores)
        running_caption = self.tokenizer.decode(beam_input_ids)
        #print('running caption:', running_caption)

        dmm_loss = self.dmm.dmm_handler(running_caption, self.tags, calc_dmm_loss=True)
        #print('dmm loss:', dmm_loss)

        dmm_component = (self.alpha * (1-hist_conf[beam_index]) * dmm_loss)
        #dmm_component = (self.alpha * dmm_loss)
        dmm_component = self.normalize_dmm(dmm_component)
        #print('dmm component:', dmm_component)

        snt_component = (1-self.alpha) * hist_conf[beam_index] * (-1*beam_scores)
        #snt_component = (1-self.alpha) * (-1*beam_scores)
        snt_component = self.normalize_lm(snt_component)
        #print('snt component:', snt_component)
        #print('------------------------------------------')

        final_scores_tensor = -1 * (dmm_component + snt_component)

        scores[beam_index] = final_scores_tensor
        #sorted_ = final_scores_tensor.sort().values
        #print('final scores:', scores)
    
    return scores