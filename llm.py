import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

class LLM():
    def __init__(self, model:str, low_cpu_mem:bool=True, window_size:int=50):
            self.model_name = model
            self.device = self._set_device()
            self.low_cpu_mem = low_cpu_mem
            self.tokenizer = self._set_tokenizer()
            self.model = self._set_model()
            self.window_size = window_size

    def _set_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')


    def _set_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _set_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, low_cpu_mem_usage=self.low_cpu_mem).to(self.device)
        model.config.pad_token_id = model.config.eos_token_id
        model.eval()
        print('model loaded: ' + self.model_name)

        return model


    def calculate_perplexity(self, input_sentence:str):
        tokenized = self.tokenizer(input_sentence)
        input = torch.tensor(tokenized.input_ids).to(self.device)
        with torch.no_grad():
            output = self.model(input, labels=input)
        
        return np.log(np.asarray(torch.exp(output.loss).cpu()))

    def calculate_perplexity_sliding(self, input_sentence:str):
        tokenized = self.tokenizer(input_sentence)
        input = torch.tensor(tokenized.input_ids).to(self.device)
        min_perplexity = 100000
        with torch.no_grad():
            for start_idx in range(input.shape[0]-self.window_size):
                input_window = input[start_idx: start_idx+self.window_size]
                output = self.model(input_window, labels=input_window)
                min_perplexity = min(min_perplexity, np.asarray(torch.exp(output.loss).cpu()))
        return np.log(min_perplexity)
        