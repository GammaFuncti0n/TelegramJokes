import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class LSTMModule():
    def __init__(self, config, tokenizer):
        '''
        Init method
        '''
        self.config = config
        self.tokenizer = tokenizer

        self.model_parameters = self.config['model']['model_params']
        self.train_parameters = self.config['train_params']
        self.device = self.config['env']['device']
        self.__init_model()

        self._criterion = nn.CrossEntropyLoss(ignore_index=0)

        lr = float(self.train_parameters['lr'])
        weight_decay = float(self.train_parameters['weight_decay'])
        gamma = float(self.train_parameters['gamma'])
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, gamma=gamma)
        self.scheduler_step = self.train_parameters['scheduler_step']

        self.num_epochs = self.train_parameters['num_epochs']
        self.clip_grad = self.train_parameters['gradient_clip']
        self.loss_logging_step = self.train_parameters['loss_logging_step']

    def __init_model(self):
        '''
        Initialize model with params from config
        '''
        self._model = LSTMModel(
            **self.model_parameters
        ).to(self.device)

    def fit(self, train_dataloader, val_dataloader=None):
        '''
        Fit model
        '''
        best_loss = np.inf
        for self.epoch in range(self.num_epochs):
            train_loss = self.train_epoch(train_dataloader)
            if self.epoch%self.scheduler_step==0:
                self._scheduler.step()
                logger.info(self.generate("Заходит улитка в бар,"))
            logger.info(f"{self.epoch+1}/{self.num_epochs}: Train loss = {train_loss:.4f}, lr = {self._scheduler.get_last_lr()[0]:.6f}")

            # Save best model
            if(train_loss < best_loss):
                best_loss = train_loss

                directory = self.config['paths']['models_checkpoints']
                os.makedirs(directory, exist_ok=True)
                torch.save(
                    {
                        'epoch': self.epoch,
                        'model_state_dict': self._model.state_dict(),
                        'loss': best_loss,
                        'model_params': self.model_parameters,
                        'training_params': self.train_parameters,
                    }, 
                    os.path.join(directory, f"{self.config['model']['name']}.pt")
                    )
    
    def train_epoch(self, train_dataloader):
        train_loss_list = []
        self._model.train()

        pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {self.epoch+1}/{self.num_epochs}', postfix={'loss': '?'}) 
        for i, batch in enumerate(train_dataloader):
            x = batch[0][:,:-1].to(self.device)
            y = batch[0][:,1:].to(self.device)

            out = self._model(x)

            self._optimizer.zero_grad()
            loss = self._criterion(out.reshape(-1, out.shape[-1]), y.reshape(-1))
            train_loss_list.append(loss.item())
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            self._optimizer.step()

            if (i+1)%self.loss_logging_step==0:
                logger.info(f"Loss batch {i+1}: {np.mean(train_loss_list[-self.loss_logging_step:])}")

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            pbar.update(1)
        pbar.close()

        return np.mean(train_loss_list)
    
    def generate(self, input_text):
        input_tokens = self.tokenizer.encode(input_text)

        input_tokens = torch.tensor(input_tokens[:-1], dtype=torch.long).unsqueeze(0).to(self.device)

        temperature = 0.5
        for i in range(100):
            self._model.eval()
            model_out = self._model(input_tokens).squeeze(0)
            model_p = torch.softmax(model_out/temperature, 1)
            sample_tokens = torch.multinomial(model_p[-1], 1)
            #next_token = model(input_tokens).argmax(-1).squeeze(0)[-1]
            input_tokens = torch.concat((input_tokens, sample_tokens.unsqueeze(0)), 1)
            if input_tokens[0,-1]==3:
                break

        output_tokens = input_tokens.squeeze(0).cpu()
        output_text = self.tokenizer.decode(list(output_tokens))
        return output_text

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout, **kwargs):
        super(LSTMModel, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size
        )
        self.encoder = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout
            )

        self.head = nn.Linear(hidden_size, self.vocab_size)
    def forward(self, x):
        emb = self.embeddings(x)
        out, (h, c) = self.encoder(emb)
        out = self.head(out)
        return out

