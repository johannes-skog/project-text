from typing import (
    Any,
    List,
    Dict,
    Callable
)

import numpy as np
import pandas as pd
import logging
from dataset.tokenizer import Tokenizer
import pyarrow as pa
from dataclasses import dataclass
from numba import jit
from dataset.text import TextFileIterator


@dataclass
class Batch(object):

    tokens: np.array = None

    text: List[str] = None

    attention: np.array = None

    def size(self):
        return self.tokens.shape[0]

    def length(self):
        return self.tokens.shape[1]


@dataclass
class MLMBatch(Batch):

    labels: np.array = None
    label_mask: np.array = None

    text_masked: List[str] = None


@jit(nopython=True)
def _search(s: int, e: int, offsets: np.array):

    assert e > s

    start_slice = (None, None)
    end_slice = (None, None)

    for i, offset in enumerate(offsets):
        if offset > s:
            break
        start_slice = (i, s - offset)

    for i, offset in enumerate(offsets[start_slice[0]:]):
        if offset > e:
            break
        end_slice = (i+start_slice[0], e - offset)

    return start_slice, end_slice


class Dataset(object):

    TOKEN_IDS_COLUMN = "token_ids"

    def __init__(
        self,
        tokenizer: Tokenizer,
        batch_size: int = 100,
        max_length: int = 512,
        debug: bool = False,
    ):

        self._log = logging.getLogger()

        self._tokenizer = tokenizer

        self._batch_size = batch_size

        self._max_length = max_length

        self._data = None

        self._batches = []
        self._offsets = []

        self._iter = None

        self._debug = debug

        self._columns_to_load = [
            self.TOKEN_IDS_COLUMN
        ]

    def set_columns_to_load(self, columns: List[str]):

        assert self.TOKEN_IDS_COLUMN in columns, "token ids must be in selected list of columns"

        self._columns_to_load = columns

    def get_columns_to_load(self):

        return self._columns_to_load

    def setup(self, file_path: str):

        with pa.memory_map(file_path, 'rb') as source:
            self._data = pa.ipc.open_file(source).read_all()

        self._batches = [recordbatch for recordbatch in self._data.to_batches()]

        self._offsets = np.cumsum([0] + [len(b) for b in self._batches], dtype=np.int64)

    def __len__(self):

        if self._data is None:
            return 0

        return len(self._data)

    def __iter__(self):

        assert self._data is not None, "setup data"

        self._iter = 0

        return self

    def _slice(self, offset: int, length: int):
        """slice with start offset and end length+offset"""

        data = {}
        """
        start_idx, end_idx = _search(offsets=self._offsets, s=offset, e=offset+length)

        # Within same batch
        if start_idx[0] == end_idx[0]:

            batches = [self._batches[start_idx[0]]]

            batches[0] = batches[0].slice(
                offset=start_idx[1],
                length=end_idx[1]
            )

        else:

            batches = self._batches[start_idx[0]:end_idx[0]+1]

            batches[0] = batches[0].slice(
                offset=start_idx[1],
                length=len(batches[0])
            )

            batches[-1] = batches[-1].slice(
                offset=0,
                length=end_idx[1]
            )

        batch_df = pa.Table.from_batches(batches)

        """

        for col in self._columns_to_load:
            data[col] = self._data.slice(offset, length)[col].to_pandas().values

        return data

    @staticmethod
    def _trim_select_pad(arrays: List[np.array], max_length: int, pad_id: int):

        n = len(arrays)
        _m = max([len(arrays[i]) for i in range(n)])
        m = _m if _m < max_length else max_length

        tokens = np.ones((n, m), dtype=np.int32) * pad_id

        for i in range(n):

            nn = len(arrays[i])

            if nn <= max_length:

                s, e = 0, nn

            else:

                s = np.random.randint(0, nn - max_length)
                e = s + max_length

            tokens[i, 0:(e-s)] = arrays[i][s:e]

        return tokens

    @staticmethod
    def _generate_attention_mask(tokens: np.array, pad_id: int):

        attention = np.ones_like(tokens)

        mask = (tokens == pad_id)

        attention[mask] = 0

        return attention

    def _process_batch(self, batch: Dict):

        if self._debug:

            text = self._tokenizer.decode([x.flatten().tolist() for x in batch[self.TOKEN_IDS_COLUMN]])

        else:

            text = None

        tokens = self._trim_select_pad(
            arrays=batch[self.TOKEN_IDS_COLUMN],
            max_length=self._max_length,
            pad_id=self._tokenizer.pad_token[1],
        )

        generic_batch = Batch(
            tokens=tokens,
            attention=self._generate_attention_mask(tokens=tokens, pad_id=self._tokenizer.pad_token[1]),
            text=text,
        )

        return generic_batch

    def __next__(self):

        if self._iter >= len(self):
            raise StopIteration

        s = self._iter
        e = min(s + self._batch_size, len(self))
        length = e-s

        data = self._slice(
            offset=s, length=length
        )

        data = self._process_batch(data)

        self._iter += length

        return data

    def split(self, file_path: str, indicies: List[int], chunk_size: int = 100):

        def _selection(ilist: List[int]):

            batch_indices = np.searchsorted(self._offsets, ilist, side="right") - 1

            batch = pa.Table.from_batches(
                [
                    self._batches[batch_idx].slice(i - self._offsets[batch_idx], 1)
                    for batch_idx, i in zip(batch_indices, ilist)
                ],
            )

            return batch.to_pandas()

        chunks = int(len(indicies) / chunk_size)

        # Sort them, so that the chunks are within the batches
        indicies.sort()

        indicies_lists = np.array_split(
            np.array(indicies),
            chunks
         )

        self._to_arrow_file(
            file_path=file_path,
            schema=self._data.schema,
            iterator=indicies_lists,
            preprocess_func=_selection,
            iter_log=int(10000/chunk_size)
        )

    def _to_arrow_file(
        self,
        file_path: str,
        schema: Any,
        iterator: Any,
        iter_log: int = 10000,
        preprocess_func: Callable = None,
    ):

        iter = 0

        with pa.OSFile(file_path, 'wb') as sink:

            with pa.ipc.new_file(sink, schema) as writer:

                for sub in iterator:

                    if sub is None:
                        break

                    if iter % iter_log == 0:
                        self._log.info(f"Process entry {iter}")

                    writer.write(
                        pa.record_batch(
                            preprocess_func(sub) if preprocess_func is not None else sub,
                            schema
                        )
                    )

                    iter += 1


class MLMdataset(Dataset):

    def __init__(
        self,
        tokenizer: Tokenizer,
        batch_size: int = 100,
        max_length: int = 512,
        mask_probability: float = 0.15,
        retain_mask_probability: float = 0.8,
        convert_mask_probability: float = 0.1,
        debug: bool = False,
    ):

        super().__init__(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            debug=debug,
        )

        self._mask_probability = mask_probability
        self._retain_mask_probability = retain_mask_probability
        self._convert_mask_probability = convert_mask_probability

    def text_file_to_arrow(self, iterator: TextFileIterator, file_path: str):

        def _package_for_file(data: List[List[str]]):

            tokens = self._tokenizer.encode(data)

            df = pd.DataFrame()
            df[self.TOKEN_IDS_COLUMN] = tokens

            return df

        self._to_arrow_file(
            file_path=file_path,
            schema=pa.Schema.from_pandas(
                _package_for_file(["test", "example"])
            ),
            iterator=iterator,
            iter_log=int(10000 / iterator._batch_size),
            preprocess_func=_package_for_file,
        )

        return file_path

    def mask(self, batch: Batch, ignore_token_ids: List[int] = []):

        # Create unifom rand for each token.
        U1 = np.random.rand(batch.size(), batch.length())
        U2 = np.random.rand(batch.size(), batch.length())

        labels = batch.tokens.copy()
        masked_tokens = batch.tokens.copy()

        # We will mask these
        label_mask = (U1 > (1 - self._mask_probability))

        mask_ignore = np.isin(batch.tokens, ignore_token_ids)

        label_mask[mask_ignore] = False

        mask_masking_retain = label_mask & (U2 > (1 - self._retain_mask_probability))

        mask_masking_convert = label_mask & (U2 > (1 - self._convert_mask_probability))

        # Replace token with mask token
        masked_tokens[mask_masking_retain] = self._tokenizer.mask_token[1]

        random_tokens = np.random.randint(
            low=0,
            high=len(self._tokenizer),
            size=(mask_masking_convert.sum())
        )

        # Replace token with random token
        masked_tokens[mask_masking_convert] = random_tokens

        return masked_tokens, labels, label_mask

    def _process_batch(self, batch: Dict):

        generic_batch = super()._process_batch(batch)

        masked_tokens, labels, label_mask = self.mask(
            batch=generic_batch,
            ignore_token_ids=[
                self._tokenizer.unk_token[1],
                self._tokenizer.eos_token[1],
                self._tokenizer.bos_token[1],
                self._tokenizer.pad_token[1],
            ],
        )

        if self._debug:

            text_masked = self._tokenizer.decode(
                [x.flatten().tolist() for x in np.array_split(masked_tokens, generic_batch.size(), axis=0)]
            )

        else:

            text_masked = None

        mlm_batch = MLMBatch(
            tokens=masked_tokens,
            attention=generic_batch.attention,
            label_mask=label_mask,
            labels=labels,
            text=generic_batch.text,
            text_masked=text_masked
        )

        return mlm_batch
