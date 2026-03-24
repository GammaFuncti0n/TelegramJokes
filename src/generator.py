import os
import torch
from .models import LSTMModel, JokesTokenizer, TransformerModel
import logging

class LSTMGenerator():
    '''
    Class for generate text
    '''
    def __init__(self, config):
        self.config = config

        self.model_path = os.path.join(self.config['paths']['model_path'], self.config['model']['model_name'])
        self.tokenizer_path = os.path.join(self.config['paths']['tokenizer_path'], self.config['model']['tokenizer_name'])
        self.device = self.config['env']['device']
        self.vocab_size = self.config['model']['vocab_size']
        self.special_tokens = self.config['model']['special_tokens']

        self.eos_token = 3

        self.__load_model()
        self.__load_tokenizer()

    def __load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model = LSTMModel(**checkpoint['model_params'])
        assert self.vocab_size == checkpoint['model_params']['vocab_size'], "vocab_size in config and checkpoint does not match."
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def __load_tokenizer(self):
        self.tokenizer = JokesTokenizer(self.vocab_size, self.special_tokens)
        self.tokenizer.load(self.tokenizer_path)

    @torch.no_grad()
    def generate(self, prompt="", maxlen=128, temperature=0.5):
        '''
        Method for generate text from scratch or from prompt
        '''
        tokens = self.tokenizer.encode(prompt)[:maxlen]
        generated = tokens[:-1]
        input_tokens = torch.tensor(generated, dtype=torch.long).unsqueeze(0)

        h, c = None, None
        for i in range(maxlen):
            emb = self.model.embeddings(input_tokens)
            if h is None:
                out, (h,c) = self.model.encoder(emb)
            else:
                out, (h,c) = self.model.encoder(emb, (h, c))
            logits = self.model.head(out)[:,-1] / temperature
            probs = torch.softmax(logits, -1)
            input_tokens = torch.multinomial(probs[-1], 1).unsqueeze(0)
            generated.append(input_tokens.item())
            if generated[-1]==self.eos_token:
                break
        output_text = self.tokenizer.decode(generated)
        return output_text

class TransformerGenerator():
    '''
    Class for generate text
    '''
    def __init__(self, config):
        self.config = config

        self.model_path = os.path.join(self.config['paths']['model_path'], self.config['model']['model_name'])
        self.tokenizer_path = os.path.join(self.config['paths']['tokenizer_path'], self.config['model']['tokenizer_name'])
        self.device = self.config['env']['device']
        self.vocab_size = self.config['model']['vocab_size']
        self.special_tokens = self.config['model']['special_tokens']

        self.eos_token = 3

        self.__load_model()
        self.__load_tokenizer()

    def __load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model = TransformerModel(**checkpoint['model_params'])
        assert self.vocab_size == checkpoint['model_params']['vocab_size'], "vocab_size in config and checkpoint does not match."
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def __load_tokenizer(self):
        self.tokenizer = JokesTokenizer(self.vocab_size, self.special_tokens)
        self.tokenizer.load(self.tokenizer_path)

    @torch.no_grad()
    def generate(self, prompt="", maxlen=128, temperature=0.5):
        input_tokens = self.tokenizer.encode(prompt)[:maxlen]
        input_len = len(input_tokens)
        input_tokens = torch.tensor(input_tokens[:-1], dtype=torch.long).unsqueeze(0).to(self.device)

        for i in range(maxlen-input_len):
            model_out = self.model(input_tokens).squeeze(0)
            model_p = torch.softmax(model_out/temperature, 1)
            sample_tokens = torch.multinomial(model_p[-1], 1)
            #next_token = model(input_tokens).argmax(-1).squeeze(0)[-1]
            input_tokens = torch.concat((input_tokens, sample_tokens.unsqueeze(0)), 1)
            if input_tokens[0,-1]==3:
                break

        output_tokens = input_tokens.squeeze(0).cpu()
        output_text = self.tokenizer.decode(list(output_tokens))
        return output_text
    
class RAGJoke():
    '''
    Class for retrieve jokes from joke base
    '''
    def __init__(self, config):
        self.config = config
    
    @torch.no_grad()
    def generate(self, prompt):
        return None