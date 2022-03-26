import torch
from torch import nn
import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        """
        self.bot = GPT3_api or Blenderbot or DialogGPT
        """
        self.device = config.device
        self.tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
        self.lm.to(self.device)
        self.lm.eval()

    def make_response(self,prefix_sentences, prompts):
        m = []
        with torch.no_grad():
            sentences = []
            for i in range(len(prompts)):
                sentences.append((prompts[i] + prefix_sentences[i]))
            reply_string = []
            eos = [self.tokenizer.encoder["<|endoftext|>"]]

            sentences_tmp = []
            for i in range(len(sentences)):
                tmp = self.tokenizer.encode(sentences[i], add_prefix_space=True)
                sentences_tmp.append(list(tmp))
            sentences = sentences_tmp

            for i in range(len(sentences)):
                temp_m = [1 for x in range(len(sentences[i]))]
                m.append(temp_m[:])

            # prepare original input to model
            prev_input = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in sentences], value=0)).to(self.device)

            m = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in m], value=0)).to(self.device)
            # take out the hidden state of original input and form it as past
            position_ids = m.long().cumsum(-1) - 1 
            position_ids.masked_fill_(m == 0, 1).to(self.device)
            _, past = self.lm(prev_input, past=None, attention_mask=m, position_ids=position_ids)

            # append eos token in the end (add attention mask 1 in the eos)
            prev_input = torch.LongTensor([[eos] * len(sentences)]).squeeze(0).to(self.device)
            append = torch.tensor([[1] for i in range(len(sentences))]).to(self.device)
            m = torch.cat((m, append), 1)
            position_ids = m.long().cumsum(-1) - 1
            position_ids.masked_fill_(m == 0, 1)
            position_ids = position_ids[:, -1].unsqueeze(-1).to(self.device)
            temp_sen = [[] for i in range(len(sentences))]

            for i in range(128):
                prev_input, past = self.lm(prev_input, past=past, attention_mask=m, position_ids=position_ids)
                m = torch.cat((m, append), 1)
                position_ids = m.long().cumsum(-1) - 1
                position_ids.masked_fill_(m == 0, 1)
                position_ids = position_ids[:, -1].unsqueeze(-1).to(self.device)

                prev_input = prev_input.squeeze(0).squeeze(1)
                prev_input = prev_input / 0.7
                prev_input = torch.softmax(prev_input[:, :50257], dim=-1)

            # prev_input = torch.multinomial(prev_input, num_samples=1)
                prev_input = torch.argmax(prev_input, dim=-1)[:, None]

                if i == 0:
                    for j in range(len(sentences)):    
                        temp_sen[j].append(prev_input[j].item())
                    continue
                flag = 1
                for j in range(len(sentences)):
                    if temp_sen[j][-1] != eos[0]: 
                        flag = 0
                        temp_sen[j].append(prev_input[j].item())
                if flag == 1: break
            a = [[self.tokenizer.decode(x).replace('<|endoftext|>', '')] for x in temp_sen]
        return a
