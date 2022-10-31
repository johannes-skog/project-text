from typing import (
    Any,
    Callable,
    Dict
)
import pandas as pd
import numpy as np
import torch
import math
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from dataset.tokenizer import Tokenizer


class Embedder(torch.nn.Module):

    def __init__(self, vocab_sz: int, hidden_dim: int):

        super().__init__()

        self._hidden_dim = hidden_dim

        self._model = torch.nn.Embedding(vocab_sz, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """

        return self._model(x)


class PositionalEncoding(torch.nn.Module):

    def __init__(self, hidden_dim: int, dropout: float = 0.1, max_seq_len: int = 512):

        super().__init__()

        self._dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))

        pos_enc = torch.zeros(1, max_seq_len, hidden_dim)

        pos_enc[:, :, 0::2] = torch.sin(position * div_term)
        pos_enc[:, :, 1::2] = torch.cos(position * div_term)

        # register_buffer, not part of state dict, is not going to be exposed to opti,
        # but be part of transfer to device etc..
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """

        x = x + self.pos_enc[:, :x.size(1), :].repeat(x.size(0), 1, 1)

        x = self._dropout(x)

        return x


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
    mask: torch.Tensor = None,
    dropout: torch.nn.Dropout = None
):

    # Swap the two last dims for key
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    # Attention
    if mask is not None:
        # attention of size [batch_size, seq_len] transform to match scores
        mask = mask.unsqueeze(1).unsqueeze(3).repeat(
            1, scores.shape[1], 1, scores.shape[3],
        )
        scores = scores.masked_fill(mask == 0, -1e9)

    # Apply softmax on the last dim
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    y = torch.matmul(scores, v)

    return y


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, heads: int, hidden_dim: int, dropout: float = 0.1):

        super().__init__()

        self._hidden_dim = hidden_dim

        self._head_dim = hidden_dim // heads

        self._heads = heads

        self._dropout = torch.nn.Dropout(dropout)

        self._q_layer = torch.nn.Linear(self._hidden_dim, self._hidden_dim)
        self._k_layer = torch.nn.Linear(self._hidden_dim, self._hidden_dim)
        self._v_layer = torch.nn.Linear(self._hidden_dim, self._hidden_dim)

        self._output_layer = torch.nn.Linear(self._hidden_dim, self._hidden_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ):

        bsz = q.size(0)

        # forward though fully-connected and split into h heads
        q = self._q_layer(q).view(bsz, -1, self._heads, self._head_dim)
        k = self._k_layer(k).view(bsz, -1, self._heads, self._head_dim)
        v = self._v_layer(v).view(bsz, -1, self._heads, self._head_dim)

        # transpose to go from dimensions [batch_size, seq_length, head, hidden_dim]
        # to [batch_size, head, seq_length, hidden_dim]

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(q, k, v, self._head_dim, mask, self._dropout)

        # concatenate heads and put through final fully-connected layer
        y = scores.transpose(1, 2).contiguous().view(bsz, -1, self._hidden_dim)

        y = self._output_layer(y)

        return y


class FullyConnected(torch.nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        internal_dim: int,
        dropout: float = 0.1,
        activation: Any = F.gelu,
    ):

        super().__init__()

        self._layer1 = torch.nn.Linear(hidden_dim, internal_dim)

        self._activation = activation

        self._dropout = torch.nn.Dropout(dropout)

        self._layer2 = torch.nn.Linear(internal_dim, hidden_dim)

    def forward(self, x: torch.Tensor):

        x = self._layer1(x)

        x = self._activation(x)

        x = self._dropout(x)

        x = self._layer2(x)

        return x


class LayerNorm(torch.nn.Module):

    def __init__(self, hidden_dim: int, epsilon: float = 1e-6):

        super().__init__()

        self._hidden_dim = hidden_dim

        self._alpha = torch.nn.Parameter(torch.ones(self._hidden_dim))

        self._beta = torch.nn.Parameter(torch.zeros(self._hidden_dim))

        self._epsilon = epsilon

    def forward(self, x: torch.Tensor):

        sigma = (x.std(dim=-1, keepdim=True) + self._epsilon)

        offset = x - x.mean(dim=-1, keepdim=True)

        y = self._alpha * ((offset / sigma) + self._beta)

        return y


class EncoderLayer(torch.nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        internal_dim: int,
        heads: int,
        dropout: float = 0.1,
        activation: Any = F.gelu,
    ):

        super().__init__()

        self._activation = activation

        self._attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
        )

        self._dropout1 = torch.nn.Dropout(dropout)

        self._layernorm1 = LayerNorm(hidden_dim)

        self._fully_connected = FullyConnected(
            hidden_dim=hidden_dim,
            internal_dim=internal_dim,
            activation=activation,
            dropout=dropout,
        )

        self._dropout2 = torch.nn.Dropout(dropout)

        self._layernorm2 = LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):

        x1 = self._layernorm1(x + self._dropout1(self._attention(k=x, v=x, q=x, mask=mask)))

        x2 = self._layernorm2(x1 + self._dropout2(self._fully_connected(x1)))

        return x2


class Encoder(torch.nn.Module):

    def __init__(
        self,
        vocab_sz: int,
        hidden_dim: int,
        internal_dim: int,
        n_encoders: int,
        heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        activation: Any = F.gelu,
    ):

        super().__init__()

        self._n_encoders = n_encoders

        self._embedder = Embedder(vocab_sz, hidden_dim)

        self._pos_enc = PositionalEncoding(hidden_dim, dropout=dropout, max_seq_len=max_seq_len)

        self._encoder_layers = torch.nn.ModuleList(
            [
                EncoderLayer(
                    hidden_dim=hidden_dim,
                    internal_dim=internal_dim,
                    heads=heads,
                    dropout=dropout,
                    activation=activation,
                )
                for x in range(self._n_encoders)
            ]
        )

        self._layernorm = LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):

        x = self._embedder(x)

        x = self._pos_enc(x)

        for i in range(self._n_encoders):

            x = self._encoder_layers[i](x, mask)

        return self._layernorm(x)


class Model(torch.nn.Module):

    MODEL_CKP_PREFIX = "models"
    OPT_CKP_PREFIX = "opts"
    META_CKP_PREFIX = "meta"

    def __init__(self):

        super().__init__()

        self._models = {}
        self._optimizers = {}

    def register_optimzier(self, name: str, optimizer: torch.nn.Module = optim.Adam, optimizer_params: Dict = {}): # noqa

        assert name in self._models, "you must register the model"

        self._optimizers[name] = optimizer(params=self._models[name].parameters(), **optimizer_params)

    @property
    def device(self):

        return next(self.parameters()).device

    def register_model(
        self,
        name: str,
        model: torch.nn.Module,
        optimize: bool = True,
        optimizer_params: Dict = {},
    ):

        self._models[name] = model

        if optimize:
            self.register_optimzier(name=name, optimizer_params=optimizer_params)

    def checkpoint(self, path: str, meta: Dict[str, Any] = None):

        data = {
            self.MODEL_CKP_PREFIX: {},
            self.OPT_CKP_PREFIX: {},
            self.META_CKP_PREFIX: {}
        }

        for n in self._models.keys():
            data[self.MODEL_CKP_PREFIX][n] = self._models[n].state_dict()

        for n in self._optimizers.keys():
            data[self.OPT_CKP_PREFIX][n] = self._optimizers[n].state_dict()

        if meta is not None:
            data[self.META_CKP_PREFIX].update(meta)

        torch.save(data, path)

        return path

    def load(self, path: str):

        data = {}

        data = torch.load(path)

        if self.MODEL_CKP_PREFIX in data:
            for n in self._models.keys():
                if n in data[self.MODEL_CKP_PREFIX]:
                    print(f"load model {n}")
                    self._models[n].load_state_dict(data[self.MODEL_CKP_PREFIX][n])

        if self.OPT_CKP_PREFIX in data:
            for n in self._optimizers.keys():
                if n in data[self.OPT_CKP_PREFIX]:
                    self._optimizers[n].load_state_dict(data[self.OPT_CKP_PREFIX][n])

        if self.META_CKP_PREFIX in data:
            return data[self.META_CKP_PREFIX]
        else:
            return {}

    def to(self, device: str):

        for n in self._models.keys():
            self._models[n].to(device)


class MLMhead(torch.nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        activation: Any = F.elu,
    ):

        super().__init__()

        self._activation = activation

        self._layer1 = torch.nn.Linear(input_dim, input_dim)

        self._layernorm1 = LayerNorm(input_dim)

        self._dropout = torch.nn.Dropout(dropout)

        self._layer2 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor):

        x = self._layer1(x)
        x = self._layernorm1(x)
        x = self._activation(x)
        x = self._dropout(x)
        x = self._layer2(x)

        return x


class MLMtransformer(Model):

    IGNORE_INDEX = -1

    def __init__(
        self,
        vocab_sz: int,
        hidden_dim: int,
        internal_dim: int,
        n_encoders: int,
        heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        activation: Any = F.elu,
        optimize: bool = True,
        optimizer_params: Dict = {},
    ):

        super().__init__()

        self._vocab_sz = vocab_sz

        self._encoder = Encoder(
            vocab_sz=vocab_sz,
            hidden_dim=hidden_dim,
            internal_dim=internal_dim,
            n_encoders=n_encoders,
            max_seq_len=max_seq_len,
            heads=heads,
            dropout=dropout,
            activation=activation,
        )

        self.register_model(
            name="encoder",
            model=self._encoder,
            optimize=optimize,
            optimizer_params=optimizer_params,
        )

        self._mlm_head = MLMhead(
            input_dim=hidden_dim,
            output_dim=vocab_sz,
            dropout=dropout,
            activation=activation,
        )

        self.register_model(
            name="head",
            model=self._mlm_head,
            optimize=optimize,
            optimizer_params=optimizer_params,
        )

        self._loss_fct = CrossEntropyLoss(
            ignore_index=self.IGNORE_INDEX,
            reduction="mean",
        )

    def loss(self, model_out: torch.Tensor, labels: torch.Tensor, label_masks: torch.Tensor = None):

        labels[label_masks == False] = self.IGNORE_INDEX # noqa

        _loss = self._loss_fct(model_out.view(-1, self._vocab_sz), labels.view(-1))

        return _loss

    def _accuracy(self, model_out: torch.Tensor, labels: torch.Tensor, label_masks: torch.Tensor = None):

        n_correct = (
            (labels == model_out.argmax(axis=2)) & (label_masks == 1)
        ).sum().item()

        n_total = label_masks.sum().item()

        accuracy = n_correct / n_total

        return accuracy

    def optimize(
        self,
        tokens: torch.Tensor,
        attention: torch.Tensor,
        labels: torch.Tensor,
        label_masks: torch.Tensor,
        zero_grad: bool = True,
        optimize_step: bool = True,
        grad_norm_clipping: Dict[str, int] = None,
    ):

        self.train()

        if zero_grad:
            for n in self._models.keys():
                self._models[n].zero_grad()

        y = self.forward(tokens=tokens, attention=attention)

        loss = self.loss(
            model_out=y,
            labels=labels,
            label_masks=label_masks
        )

        accuracy = self._accuracy(
            model_out=y,
            labels=labels,
            label_masks=label_masks
        )

        loss.backward()

        if grad_norm_clipping is not None:
            grad_norms = {}
            for n in self._models.keys():
                grad_norms["Grad_norm_" + n] = torch.nn.utils.clip_grad_norm_(
                    self._models[n].parameters(),
                    grad_norm_clipping[n],
                ).item()

        if optimize_step:

            for n in self._optimizers.keys():
                self._optimizers[n].step()

        metrics = {
            "Loss": loss.item(),
            "Accuracy": accuracy,
        }

        metrics.update(grad_norms)

        return metrics

    def forward(self, tokens: torch.Tensor, attention: torch.Tensor):

        x = self._encoder(x=tokens, mask=attention)

        y = self._mlm_head(x)

        return y

    def forward_debug(self, text: str, tokenizer: Tokenizer, topn: int = None):

        self.train(False)

        tokens_ids = tokenizer.encode(
            list=[text],
            out_type=int,
            preprocess=True,
        )

        tokens_ids = np.expand_dims(
            np.concatenate(tokens_ids),
            axis=0,
        )

        probs = self.forward(
            tokens=torch.IntTensor(tokens_ids).to(self.device),
            attention=torch.IntTensor(np.ones_like(tokens_ids)).to(self.device),
        ).detach()

        max_tokens = torch.argmax(probs, dim=2)[0]

        print(len(max_tokens))

        print((tokens_ids.flatten() == max_tokens.cpu().numpy()).mean())

        output_str = tokenizer.decode([int(x) for x in list(max_tokens.flatten())])

        print(tokenizer.decode([int(x) for x in list(tokens_ids.flatten())]))
        print(output_str)

        ids = np.where(tokens_ids.flatten() == tokenizer.mask_token[1])[0]

        softmax = torch.nn.Softmax(dim=0)

        topn = topn if topn is not None else len(tokenizer)

        dfs = []

        for mask_pos in ids:

            print(mask_pos)

            proba = softmax(probs[0, mask_pos, :]).cpu().numpy()[0:len(tokenizer)]

            df = pd.DataFrame()

            df["token"] = [tokenizer.decode([x]) for x in range(len(tokenizer))]
            df["proba"] = proba
            df["mask_pos"] = mask_pos

            dfs.append(df.sort_values("proba", ascending=False).head(topn))

        dfs = pd.concat(dfs)

        return dfs, output_str
