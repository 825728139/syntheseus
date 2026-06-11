# SharedMemory + numpy 内存共享改造计划

## Context

当前 server.py CPU 模式使用 fork+COW，但 Python 对象在子进程访问时触发 refcount 写 → COW 页复制。3 进程各 13.1 GB（共 ~39 GB）。改造目标：降至 ~5 GB/进程，共 ~5 GB 物理内存。

**只支持 fork（CPU）模式**，spawn（GPU）模式保持现状。

## 改造文件

| 文件 | 改动 |
|---|---|
| `syntheseus/reaction_prediction/inference/shared_state.py` | **新建** — SharedMemory 封装类 |
| `syntheseus/reaction_prediction/inference/simpretro.py` | 改造 __init__（lines 187-208）、_match_single_molecule（lines 63-83） |

worker.py 和 server.py **不需要改动**——fork 模式下子进程通过 fork 继承整个模型对象（包括 SharedMemory 引用），无需额外 attach 逻辑。

## 实现步骤

### Step 1: 新建 shared_state.py

路径：`syntheseus/reaction_prediction/inference/shared_state.py`

```python
"""SharedMemory wrappers for cross-process data sharing (fork mode)."""
import atexit
import numpy as np
import torch
from multiprocessing.shared_memory import SharedMemory


class SharedArray:
    """Numpy array backed by POSIX shared memory.

    Main process: create=True, writes data.
    Fork children: inherit the object via fork, no attach needed.
    """

    def __init__(self, name: str, shape: tuple, dtype: np.dtype,
                 create: bool = False, data: np.ndarray = None):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        self.name = name
        self.shm = SharedMemory(name=name, create=create, size=nbytes)
        self.arr = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        if create and data is not None:
            self.arr[:] = data
            atexit.register(self._cleanup)

    def _cleanup(self):
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception:
            pass


class SharedWeights:
    """PyTorch model weights in shared memory (float32 contiguous buffer).

    Layout: all parameters flattened and concatenated.
    Subprocess restores via torch.from_numpy (zero-copy).
    """

    # Parameter name -> shape, in load order
    PARAM_LAYOUT = [
        ('fc1.weight', (2048, 8192)),
        ('fc1.bias',   (2048,)),
        ('bn.weight',  (2048,)),
        ('bn.bias',    (2048,)),
        ('bn.running_mean', (2048,)),
        ('bn.running_var',  (2048,)),
        ('bn.num_batches_tracked', ()),
        ('fc2.weight', (1, 2048)),
        ('fc2.bias',   (1,)),
    ]

    def __init__(self, name: str, create: bool = False, model=None):
        total_floats = sum(int(np.prod(s)) for _, s in self.PARAM_LAYOUT)
        nbytes = total_floats * 4  # float32
        self.name = name
        self.shm = SharedMemory(name=name, create=create, size=nbytes)
        self.buf = np.ndarray((total_floats,), dtype=np.float32, buffer=self.shm.buf)

        if create and model is not None:
            offset = 0
            sd = model.state_dict()
            for key, shape in self.PARAM_LAYOUT:
                n = int(np.prod(shape))
                self.buf[offset:offset+n] = sd[key].flatten().numpy()
                offset += n
            atexit.register(self._cleanup)

    def load_into(self, model):
        """Zero-copy: restore weights from shared buffer into model."""
        offset = 0
        state = {}
        for key, shape in self.PARAM_LAYOUT:
            n = int(np.prod(shape))
            view = self.buf[offset:offset+n].reshape(shape)
            state[key] = torch.from_numpy(view)  # zero-copy tensor
            offset += n
        model.load_state_dict(state, assign=True)

    def _cleanup(self):
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception:
            pass
```

### Step 2: 改造 simpretro.py __init__

**当前代码** (lines 187-208)：
```python
self.fingerprint_base = filter_fingerprint_base
self.template_fps = []
for template in self.templates_raw:
    if template in self.fingerprint_base:
        self.template_fps.append(self.fingerprint_base[template])
    else:
        fp = smarts_to_fingerprint(template)
        self.template_fps.append(fp)
        self.fingerprint_base[template] = fp
self.template_fps = np.array(self.template_fps)
save_fingerprint_base()

self.filter = Net_orig()
with importlib.resources.files(...).joinpath("model_smoothbce.pth") as model_path:
    self.filter.load_state_dict(torch.load(str(model_path), map_location=self.device))
```

**改造为**：
```python
# --- 构建 template_fps（同现有逻辑）---
self.fingerprint_base = filter_fingerprint_base
self.template_fps = []
for template in self.templates_raw:
    if template in self.fingerprint_base:
        self.template_fps.append(self.fingerprint_base[template])
    else:
        fp = smarts_to_fingerprint(template)
        self.template_fps.append(fp)
        self.fingerprint_base[template] = fp
self.template_fps = np.array(self.template_fps)
save_fingerprint_base()

# --- dtype 压缩：int64 → uint8（指纹值为 0/1，8x 压缩）---
self.template_fps = self.template_fps.squeeze().astype(np.uint8)

# --- 放入 SharedMemory ---
self._fps_shm = SharedArray(
    name="simpretro_fps", create=True, data=self.template_fps
)

# --- 释放 fingerprint_base dict（~2.4 GB）---
self.fingerprint_base = None
import syntheseus.reaction_prediction.fast_filter.model as _fm
_fm.fingerprint_base = {}
import gc; gc.collect()

# --- 加载 filter，权重放入 SharedMemory ---
self.filter = Net_orig()
with importlib.resources.files(...).joinpath("model_smoothbce.pth") as model_path:
    self.filter.load_state_dict(torch.load(str(model_path), map_location=self.device))
self._weights_shm = SharedWeights(
    name="simpretro_weights", create=True, model=self.filter
)
```

### Step 3: 改造 _match_single_molecule

**当前代码** (lines 63-83)：
```python
valid_temp_fps = _MATCH_MODEL.template_fps[valid_template_id]       # int64
p_fp = smiles_to_fingerprint(smiles)                                # int64
data = torch.tensor(
    np.concatenate([valid_temp_fps.squeeze(),
                    np.repeat(p_fp, len(valid_temp_fps), axis=0)], axis=1),
    dtype=torch.float32,
)
pred = _MATCH_MODEL.filter(data).squeeze().cpu().numpy()
# ...
pred[valid_template_id.index(v[2])]   # O(N) linear scan
```

**改造为**：
```python
# 从 SharedMemory 读取 uint8，转 float32
valid_temp_fps = _MATCH_MODEL._fps_shm.arr[valid_template_id].astype(np.float32)
p_fp = smiles_to_fingerprint(smiles).astype(np.float32)
data = torch.tensor(
    np.concatenate([valid_temp_fps.squeeze(),
                    np.repeat(p_fp, len(valid_temp_fps), axis=0)], axis=1),
    dtype=torch.float32,
)
pred = _MATCH_MODEL.filter(data).squeeze().cpu().numpy()

# 优化：O(N) list.index() → O(1) dict 查找
idx_map = {v: i for i, v in enumerate(valid_template_id)}
# 然后用 pred[idx_map[v[2]]] 替代 pred[valid_template_id.index(v[2])]
```

## 不需要改动的文件

- **worker.py**：fork 模式下 `_MODEL` 通过 fork 继承，包含 SharedMemory 引用，无需额外逻辑
- **server.py**：`_create_pool` 中 fork 路径不变，atexit 清理由 SharedArray/SharedWeights 构造函数自动注册
- **model.py**：`fingerprint_base` 全局 dict 在 simpretro.py 中通过模块引用清空

## 内存效果预期

| 组件 | 改造前（每进程） | 改造后（每进程） |
|---|---|---|
| C++ TemplateLibrary | ~4 GB (COW 共享) | ~4 GB (COW 共享，不变) |
| template_fps | ~2.4 GB (int64, COW 退化) | **~305 MB** (uint8, SharedMemory) |
| fingerprint_base dict | ~2.4 GB (COW 退化) | **0** (已释放) |
| filter 权重 | ~64 MB (COW 退化) | **~64 MB** (SharedMemory 共享) |
| Python 运行时 | ~1 GB | ~0.5 GB |
| **ps 报告每进程** | **~13 GB** | **~5 GB** |
| **实际物理内存** | **~39 GB** | **~5 GB** |

## 验证

```bash
# 1. 启动服务
python -m uvicorn server.server:app --host 0.0.0.0 --port 8006

# 2. 检查 ps RSS
ps aux | grep uvicorn  # 每进程应从 ~13GB 降至 ~5GB

# 3. 检查真实物理内存
cat /proc/<pid>/smaps_rollup  # Shared_Clean 增大，Private_Dirty 减小

# 4. 系统级
free -h  # 总占用从 ~39GB 降至 ~5-6GB

# 5. 功能测试
curl -X POST http://localhost:8006/api/retrosynthesis/search \
  -H 'Content-Type: application/json' \
  -d '{"smiles": "CCO", "backend": "retro_star"}'

# 6. 稳定性：多次请求后内存不退化
```
