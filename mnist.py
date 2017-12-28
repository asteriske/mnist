import gzip
import numpy as np
import os
import pandas as pd
import requests

__version__ = "0.1"


class Mnist:
    """
    Base class, holds the download() and read() method.

    Parameters
    ----------

    dataset: str, 'train' or 'test'
        Which dataset to use
    """

    def __init__(self, dataset):

        if (dataset != "train") & (dataset != "test"):
            raise ValueError("Invalid dataset type, must be one of 'train','test'.")

        self._dataset = dataset
        self._filedir = '.'
        self._local_paths = []

    def download(self, destdir='.'):

        """
        Downloads the MNIST archives from http://yann.lecun.com/exdb/mnist/

        Parameters
        ----------

        destdir: str, default '.'
            Full path to where the gzip archives should be downloaded. Defaults to
            current directory.

        """

        destdir = self._filedir
        dataset = self._dataset

        local_paths = []

        if dataset == 'train':

            file_urls = ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                         "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"]
        else:

            file_urls = ["http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                         "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"]

        for url in file_urls:

            r = requests.get(url, stream=True)

            filename = url.split('/')[-1]

            local_paths.append(os.path.join(destdir, filename))

            with open(local_paths[-1], 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

        self._local_paths = local_paths

    def read(self, output="numpy"):

        """
        Reads a label and a data .gz file and returns either numpy or pandas objects. If pandas,
        the columns of the data directories are of the form 'px*' for numbers 1-784

        Parameters
        ----------
        data: str
            path to the .gz file containing the pixel data

        output: str, 'numpy' or 'pandas', defaults to 'numpy'
            return format

        Returns
        -------
            * y: labels, either 1d numpy array or pandas Series
            * x: data, either numpy matrix or pandas DataFrame

        """

        data = self._local_paths[0]
        labels = self._local_paths[1]

        if (output != "numpy") & (output != "pandas"):
            raise ValueError("Invalid output format, must be one of 'numpy','pandas'.")

        with gzip.open(labels, 'rb') as train_l:

            train_l.seek(0)
            magic_number = int.from_bytes(train_l.read(4), byteorder='big')
            num_images = int.from_bytes(train_l.read(4), byteorder='big')

            label_array = np.empty([num_images], dtype=int)

            labs = train_l.read()

            for i in range(num_images):
                label_array[i] = labs[i]

        with gzip.open(data, 'rb') as train_f:

            train_f.seek(0)
            magic_number = int.from_bytes(train_f.read(4), byteorder='big')
            num_images = int.from_bytes(train_f.read(4), byteorder='big')
            nrow = int.from_bytes(train_f.read(4), byteorder='big')
            ncol = int.from_bytes(train_f.read(4), byteorder='big')

            data_matrix = np.empty([num_images, nrow * ncol], dtype=int)

            row_idx = 0
            col_idx = 0
            i = 0
            for k in range(num_images):
                f = train_f.read(nrow * ncol)
                for b in range(nrow * ncol):
                    data_matrix[k][b] = f[b]

        if output == "numpy":

            return label_array, data_matrix

        else:

            data_col_labels = ["px{}".format(str(x)) for x in range(data_matrix.shape[1])]

            return pd.Series(label_array), pd.DataFrame(data_matrix, columns=data_col_labels)
