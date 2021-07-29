import os
import pickle
import gzip

import gin
import numpy as np

@gin.configurable(module=__name__)
class ReplayBuffer():
    def __init__(self, max_size, fifo = True):
        # when max_size is -1, it will have infinite size replay buffer
        self.ptr = 0
        self.fifo = fifo
        self.max_size= max_size
        self._data = []

    def append(self, value):
        if self.full():
            if self.fifo:
                self._data[self.ptr]= value
            else:
                self._data[np.random.choice(len(self._data))]= value
        else:
            self._data.append(value)

        self.ptr = (self.ptr + 1) % self.max_size

    def __len__(self): return len(self._data)

    def full(self): return len(self._data) == self.max_size

    def sample(self,sample_size):
        idxes = np.random.choice(len(self._data),sample_size)

        mini_batch = []
        for idx in idxes:
            mini_batch.append(self._data[idx])

        return [np.array(elems).astype(np.float32) for elems in zip(*mini_batch)]

    def epoch_iterator(self,batch_size):
        idxes = np.random.permutation(len(self._data))

        for i in range(0,len(idxes),batch_size):
            mini_batch = [self._data[idx] for idx in idxes[i:i+batch_size]]
            yield [np.array(elems).astype(np.float32) for elems in zip(*mini_batch)]

    def save(self,log_dir,fname):
        with gzip.open(os.path.join(log_dir,fname),'wb') as f:
            pickle.dump(self,f)

    @staticmethod
    def load(fname):
        with gzip.open(fname,'rb') as f:
            return pickle.load(f)
