import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from .models import LSTMModel, JokesTokenizer, TransformerModel
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import logging

system_logger = logging.getLogger("system")

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
        system_logger.info(f"Model {self.model_path} loaded")

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
        system_logger.info(f"Model {self.model_path} loaded")

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

        self.data_path = os.path.join(self.config['paths']['data_path'], self.config['rag_model']['data_name'])
        self.model_name = self.config['rag_model']['model_name']
        self.chunk_size = self.config['rag_model']['chunk_size']

        self.embeddings_path = self.config['rag_model']['embeddings_path']

        self.__load_data()
        assert self.chunk_size <= self.database_len
        self.__load_model()
    
    def __load_data(self):
        self.data_base = []
        with open(self.data_path, 'r') as f:
            for line in f:
                self.data_base.append(str(line))
        self.data_base = np.array(self.data_base)
        self.database_len = len(self.data_base)
        system_logger.info(f"Data {self.data_path} loaded and has length = {self.database_len}")

    def __load_model(self):
        self.model = SentenceTransformer(self.model_name)#, local_files_only=True)  
        system_logger.info(f"Model {self.model_name} loaded") 

        self.joke_embeddings = torch.load(self.embeddings_path)
        system_logger.info(f"Embeddings {self.embeddings_path} loaded")
        
    def generate(self, prompt):
        chunk_indeces = np.random.choice(self.database_len, size=self.chunk_size, replace=False)
        documents = self.data_base[chunk_indeces]
        document_embeddings = self.joke_embeddings[chunk_indeces].to(torch.float32)
        #document_embeddings = self.model.encode(documents, prompt_name="query")
        query_embeddings = self.model.encode(prompt, prompt_name="query")

        scores = self.model.similarity(query_embeddings, document_embeddings)
        index = scores.argmax(-1).squeeze(0)
        system_logger.info(f"{document_embeddings.shape=}, {scores.max()=}, {scores.min()=}, {index=}")
        output_text = documents[index]
        return output_text