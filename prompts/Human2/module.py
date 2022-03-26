import torch
import torch.nn as nn

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from prompts import example

class prompt(example.module.prompt):
    def __init__(self, config):
        super().__init__(config)
        """
        self.model = ""
        self.model = DialogueGPT(config)
        """
        self.config = config
        self.device = config.device
        self.model = ""
    
    def prepare_input(self, task, input_ids, mask, model):

        return None, None, None, None

    def re_padding(self, inputs_id, mask):

        return None, None, None     
    
    