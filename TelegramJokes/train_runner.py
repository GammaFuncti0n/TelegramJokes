import os

from TelegramJokes.data import load_data, JokesDataset, collate_fn
from TelegramJokes.models import JokesTokenizer, LSTMModule, TransformerModule
from torch.utils.data import DataLoader

class TrainRunner():
    '''
    Module for load dataset and fit model
    '''
    def __init__(self, config) -> None:
        self.config = config
        
        self.tokenizer_path = config['paths']['tokenizers_checkpoints']
        self.model_name = config['model']['name']

        # Tokenizers params
        self.vocab_size = config['model']['model_params']['vocab_size']
        self.special_tokens = config['model']['tokenizer_params']['special_tokens']

        # Dataloader params
        self.model_max_length = config['model']['model_params']['model_max_length']
        self.batch_size = config['train_params']['batch_size']
        self.num_workers = config['train_params']['num_workers']
    
    def run(self) -> None:
        data = load_data(self.config['paths']['data'])

        tokenizer = JokesTokenizer(self.vocab_size, self.special_tokens)
        tokenizer_file = os.path.join(self.tokenizer_path, f'tokenizer_{self.model_name}.json')
        try:
            tokenizer.load(tokenizer_file)
        except:
            tokenizer.fit(data)
            tokenizer.save(tokenizer_file)
        
        train_dataset = JokesDataset(texts=data, tokenizer=tokenizer)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=lambda batch: collate_fn(batch, model_max_length=self.model_max_length), 
            num_workers=self.num_workers
            )

        if self.config['model']['type']=='lstm':
            model = LSTMModule(self.config, tokenizer)
        elif self.config['model']['type']=='transformer':
            model = TransformerModule(self.config, tokenizer)
        else:
            raise Exception(f"Unknown model type: {self.config['model']['type']}")
        model.fit(train_dataloader)