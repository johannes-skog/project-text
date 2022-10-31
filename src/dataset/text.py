from itertools import islice


class TextFileIterator(object):

    def __init__(self, file_path: str, batch_size: int = 1):

        self._file_path = file_path
        self._batch_size = batch_size
        self.iter = 0
        self._file = None

        self._open_file()

    def _open_file(self):

        self._file = open(self._file_path, 'r')

    def _close_file(self):

        if self._file is not None:
            self._file.close()

    def __iter__(self):

        if (self._file is None) or self._file.closed:
            self._open_file()

        self.iter = 0

        return self

    def __next__(self):

        if self._file.closed:
            raise StopIteration

        iterator = islice(self._file, self._batch_size)

        data = [line for line in iterator]

        self.iter += len(data)

        if len(data) == 0:
            self._close_file()
            data = None

        return data
