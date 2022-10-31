from abc import abstractmethod
from typing import (
    List,
    Any
)
import numpy as np
import sentencepiece as spm
import re


class Tokenizer(object):

    NEW_LINE_TOKEN = "<n>"
    MASK_TOKEN = "<mask>"
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"

    def __init__(self):

        pass

    @abstractmethod
    def encode(self, list: List[str]):
        pass

    @abstractmethod
    def decode(self, list: List[str]):
        pass

    @staticmethod
    def _wrap_text(s: str):

        s = Tokenizer.BOS_TOKEN + s + Tokenizer.EOS_TOKEN

        return s

    @staticmethod
    def _cast_new_line(s: str):

        s = re.sub("\r\n|\r|\n", Tokenizer.NEW_LINE_TOKEN, s)
        return s

    def _preprocess_text(self, list: List[str]):

        for i in range(len(list)):
            list[i] = self._wrap_text(list[i])
            list[i] = self._cast_new_line(list[i])

        return list

    def get_special_tokens(self):

        token = [
            self.NEW_LINE_TOKEN,
            self.MASK_TOKEN,
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]

        return token

    def get_special_token_ids(self, token_list: List[str] = None):

        tokens = token_list if token_list is not None else self.get_special_tokens()

        # Remove the start token
        token_ids = [x[1] for x in self.encode(tokens, preprocess=False)]

        return token_ids

    @property
    def pad_token(self):

        return self.PAD_TOKEN, self.get_special_token_ids([self.PAD_TOKEN])[0]

    @property
    def mask_token(self):

        return self.MASK_TOKEN, self.get_special_token_ids([self.MASK_TOKEN])[0]

    @property
    def bos_token(self):

        return self.BOS_TOKEN, self.get_special_token_ids([self.BOS_TOKEN])[0]

    @property
    def eos_token(self):

        return self.EOS_TOKEN, self.get_special_token_ids([self.EOS_TOKEN])[0]


class SentencePiece(Tokenizer):

    def __init__(self, model_path: str):

        self._tokenizer = spm.SentencePieceProcessor(model_file='m.model')

    @staticmethod
    def train(file_path: str, out_folder: str, **kwargs):

        # https://github.com/google/sentencepiece/blob/master/doc/options.md
        spm.SentencePieceTrainer.train(
            input=file_path,
            model_prefix=f'{out_folder}/sentencepiece',
            user_defined_symbols=[
                Tokenizer.MASK_TOKEN,
                Tokenizer.PAD_TOKEN,
                Tokenizer.NEW_LINE_TOKEN,
                Tokenizer.BOS_TOKEN,
                Tokenizer.EOS_TOKEN,
            ],
            **kwargs,
        )

    def __len__(self):

        return len(self._tokenizer)

    @property
    def vocab_size(self):
        return len(self)

    def __call__(self, s: str):

        return self.encode([s], preprocess=False)

    def decode(self, list: List[int]):

        tokens = self._tokenizer.decode(list)

        return tokens

    def get_vocab(self):

        vocab = {i: self._tokenizer.IdToPiece(i) for i in range(self.vocab_size)}

        return vocab

    @property
    def unk_token(self):

        return ' ⁇ ', 0

    def encode(
        self,
        list: List[str],
        out_type: Any = int,
        enable_sampling: bool = False,
        alpha: float = 0.1,
        nbest_size: int = -1,
        preprocess: bool = True,
    ):

        if preprocess:
            list = self._preprocess_text(list)

        tokens = self._tokenizer.encode(
            list,
            out_type=out_type,
            enable_sampling=enable_sampling,
            alpha=alpha,
            nbest_size=nbest_size,
        )

        tokens = [np.array(x, dtype=np.int32) for x in tokens]

        return tokens
