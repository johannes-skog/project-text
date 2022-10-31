from typing import Callable
import math
from datasets.dataset_dict import DatasetDict
from dataset.tokenizer import Tokenizer


def _dataset_to_file(
    datasets: DatasetDict,
    file_path: str,
    text_column: str = "text",
    preprocess_func: Callable = Tokenizer._cast_new_line
):

    """Take a huggingface dataset and write to file"""

    file = open(file_path, "w")

    for dataset_name in datasets.keys():

        N = len(datasets[dataset_name])

        iter_print = math.ceil((N/100) / 2.) * 2

        print(dataset_name)

        for i, entry in enumerate(datasets[dataset_name]):

            text = preprocess_func(entry[text_column]) if preprocess_func is not None else entry[text_column]

            file.write(f'{text}\n')

            if i % iter_print == 0:
                print(f"{round(i / N, 2)} % done")

    file.close()
