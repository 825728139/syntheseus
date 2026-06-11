"""SharedMemory wrappers for cross-process data sharing (fork mode).

Main process creates SharedMemory segments; fork children inherit them
via fork (no attach logic needed). Atexit cleanup is registered by the
creator.
"""

import atexit
import logging

import numpy as np
import torch
from multiprocessing.shared_memory import SharedMemory

logger = logging.getLogger(__name__)


class SharedArray:
    """Numpy array backed by POSIX shared memory (MAP_SHARED, no COW)."""

    def __init__(self, name: str, shape: tuple = None, dtype: np.dtype = None,
                 create: bool = False, data: np.ndarray = None):
        if data is not None:
            shape = data.shape
            dtype = data.dtype
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        self.name = name
        if create:
            # Clean up stale segment from previous run
            try:
                old = SharedMemory(name=name, create=False)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
        self.shm = SharedMemory(name=name, create=create, size=nbytes)
        self.arr = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        if create and data is not None:
            self.arr[:] = data
            atexit.register(self._cleanup)
            logger.info("SharedArray '%s': created %s %s (%.1f MB)",
                        name, dtype, shape, nbytes / 1e6)

    def _cleanup(self):
        try:
            self.shm.close()
            self.shm.unlink()
            logger.info("SharedArray '%s': cleaned up", self.name)
        except Exception:
            pass


class SharedWeights:
    """PyTorch model weights in shared memory (float32 contiguous buffer).

    All parameters are flattened and concatenated into a single buffer.
    Subprocess restores via torch.from_numpy (zero-copy).
    """

    PARAM_LAYOUT = [
        ('fc1.weight', (2048, 8192)),
        ('fc1.bias', (2048,)),
        ('bn.weight', (2048,)),
        ('bn.bias', (2048,)),
        ('bn.running_mean', (2048,)),
        ('bn.running_var', (2048,)),
        ('bn.num_batches_tracked', ()),
        ('fc2.weight', (1, 2048)),
        ('fc2.bias', (1,)),
    ]

    def __init__(self, name: str, create: bool = False, model=None):
        total_floats = sum(int(np.prod(s)) for _, s in self.PARAM_LAYOUT)
        nbytes = total_floats * 4  # float32
        self.name = name
        if create:
            try:
                old = SharedMemory(name=name, create=False)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
        self.shm = SharedMemory(name=name, create=create, size=nbytes)
        self.buf = np.ndarray((total_floats,), dtype=np.float32, buffer=self.shm.buf)

        if create and model is not None:
            offset = 0
            sd = model.state_dict()
            for key, shape in self.PARAM_LAYOUT:
                n = int(np.prod(shape))
                self.buf[offset:offset + n] = sd[key].flatten().numpy()
                offset += n
            atexit.register(self._cleanup)
            logger.info("SharedWeights '%s': created (%.1f MB)", name, nbytes / 1e6)

    def load_into(self, model):
        """Zero-copy: restore weights from shared buffer into model."""
        offset = 0
        state = {}
        for key, shape in self.PARAM_LAYOUT:
            n = int(np.prod(shape))
            view = self.buf[offset:offset + n].reshape(shape)
            state[key] = torch.from_numpy(view)
            offset += n
        model.load_state_dict(state, assign=True)

    def _cleanup(self):
        try:
            self.shm.close()
            self.shm.unlink()
            logger.info("SharedWeights '%s': cleaned up", self.name)
        except Exception:
            pass
