from datasets import load_dataset
from dataset.huggingface.utils import _dataset_to_file
from dataset.tokenizer import Tokenizer


def se_oscar(file_path: str = "tmp/se_oscar.txt"):

    datasets = load_dataset('oscar', 'unshuffled_deduplicated_sv', cache_dir="data/")

    _dataset_to_file(
        datasets=datasets,
        file_path=file_path,
        text_column="text",
        preprocess_func=Tokenizer._cast_new_line,
    )

    return file_path
