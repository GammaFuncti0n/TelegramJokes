import os
from typing import List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing

import logging
logger = logging.getLogger(__name__)

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
            raise Exception("Input should be str")
            # Need to be fix
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