import os
from typing import List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
import torch
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)

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

class JokesTokenizer():
    def __init__(self, vocab_size: int, special_tokens: List[str]):
        self._is_fitted = False

        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    def __init_tokenizer__(self):
        self._tokenizer = Tokenizer(BPE())
        self._tokenizer.decoder = ByteLevelDecoder()
        self._tokenizer.pre_tokenizer = ByteLevel()

    def fit(self, data):
        self.__init_tokenizer__()
        self.trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens
        )
        self._tokenizer.train_from_iterator(data, self.trainer)
        self._tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", self._tokenizer.token_to_id("[BOS]")),
                ("[EOS]", self._tokenizer.token_to_id("[EOS]"))
            ]
        )
        self._is_fitted = True
        logger.info("Tokenizer fitted")

    def encode(self, X):
        assert self._is_fitted
        if isinstance(X, str):
            return self._tokenizer.encode(X).ids
        else:
            return self._tokenizer.encode_batch(X).ids

    def decode(self, X):
        assert self._is_fitted
        return self._tokenizer.decode(X)

    def save(self, path):
        try:
            assert self._is_fitted
            self._tokenizer.save(path)
            logger.info("Tokenizer saved in: %s", path)
        except Exception:
            logger.exception("Failed to save tokenize: %s", path)
            raise

    def load(self, path):
        try:
            self._tokenizer = Tokenizer.from_file(path)
            self._is_fitted = True
            logger.info("Tokenizer loaded from: %s", path)
        except Exception:
            logger.exception("Failed to load file: %s", path)
            raise

def load_model(weights_path, device="cpu"):
    model = LSTMModel(5000, 256, 256, 2, 0.3)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def load_tokenizer(tokenizer_path):
    tokenizer = JokesTokenizer(5000, ["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    tokenizer.load(tokenizer_path)
    return tokenizer