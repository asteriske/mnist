import gzip
import numpy as np
import os
import requests

__version__ = "0.2"


class Mnist:
    """
    Base class, holds the download() and read() method.

    Parameters
    ----------

    dir: str
        Download location for the gzip files
    download: bool
        Whether or not to download (files assumed present if False)
    """

    def __init__(self, dir='.', download=True):

        self._arrays = {}
        self._dir = dir
        self._file_urls = {'xtrain':"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                           'ytrain':"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                           'xtest':"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                           'ytest':"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"}

        self._local_paths = {}

        if download:
            self.download()


    def batch(self, batch_size, dataset):
        """
        A generator object that feeds elements of the desired dataset
        `num` rows at a time.
        """

        assert dataset in ['train', 'test', 'all'], "Invalid dataset specification"

        if dataset == 'test':

            ybatch = self.ytest
            xbatch = self.xtest

        if dataset == 'train':

            ybatch = self.ytrain
            xbatch = self.xtrain

        if dataset == 'all':

            ybatch = np.hstack((self.ytrain, self.ytest))
            xbatch = np.vstack((self.xtrain, self.xtest))

        idx = 0

        while idx <= xbatch.shape[0]:
            idx_last = idx
            idx += batch_size
            yield (ybatch[idx_last:idx], xbatch[idx_last:idx, :])


    def download(self):

        """
        Downloads the MNIST archives from http://yann.lecun.com/exdb/mnist/
        """

        if not os.path.isdir(self._dir):
            os.makedirs(self._dir)

        local_paths = {}

        for k in self._file_urls.keys():

            url = self._file_urls[k]

            r = requests.get(url, stream=True)

            filename = url.split('/')[-1]

            local_paths[k] = os.path.join(self._dir, filename)

            with open(local_paths[k], 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

        self._local_paths = local_paths


    def _access_or_read(self, val):
        """
        Returns a numpy representation if it exists, otherwise read from the file.
        """
        if val in self._arrays.keys():
            return self._arrays[val]

        else:

            assert val in self._local_paths.keys() and os.path.exists(self._local_paths[val]), "File doesn't exist!"
            self._arrays[val] = self.read(val)

            return self._arrays[val]


    def read(self, val):
        """
        Read the file in the _local_paths dict with the given key, returning a numpy array.

        Parameters
        ----------
        val: str
            key of the _local_paths dict requested to be read
        """

        file_path = self._local_paths[val]

        if 'x' in val:

            with gzip.open(file_path, 'rb') as train_f:

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

            return data_matrix

        if 'y' in val:

            with gzip.open(file_path, 'rb') as train_l:
                train_l.seek(0)
                magic_number = int.from_bytes(train_l.read(4), byteorder='big')
                num_images = int.from_bytes(train_l.read(4), byteorder='big')

                label_array = np.empty([num_images], dtype=int)

                labs = train_l.read()

                for i in range(num_images):
                    label_array[i] = labs[i]

            return label_array


    @property
    def xtrain(self):
        return self._access_or_read('xtrain')


    @property
    def xtest(self):
        return self._access_or_read('xtest')


    @property
    def ytrain(self):
        return self._access_or_read('ytrain')


    @property
    def ytest(self):
        return self._access_or_read('ytest')
