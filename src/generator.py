import os
import torch
from .models import LSTMModel, JokesTokenizer

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