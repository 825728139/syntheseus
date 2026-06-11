# SimpRetro 内存共享改造说明

## 背景

SimpRetro 服务使用 fork+COW 模式运行，主进程加载 C++ TemplateLibrary（~4 GB）和 Python 模型数据后 fork 出 worker 进程。fork 后 worker 通过 COW（Copy-on-Write）共享父进程内存页。

**问题**：Python 对象（template_fps、filter 权重、fingerprint_base dict）在访问时会修改 refcount（CPython 引用计数），触发 COW 页复制。rdchiral 的 `Reaction::run()` 会修改模板对象的原子属性，同样触发 COW。

**结果**：每个 worker 进程 RSS ~13 GB，3 进程合计 ~39 GB（`ps` 报告值）。实际物理内存 ~12 GB（`PSS` 或 `free -h` 确认）。

## 改造方案

### 1. SharedMemory 共享 template_fps

**文件**：`simpretro.py` + `shared_state.py`

**改动**：
- template_fps 从 int64 压缩为 int16（值范围 -170..1080，4x 压缩：2.4 GB → 636 MB）
- 放入 `multiprocessing.shared_memory`（POSIX shm，MAP_SHARED）
- 子进程通过 numpy 视图直接访问，无 Python refcount，不触发 COW

**收益**：template_fps 跨进程真正共享，不退化

**代价**：每次访问需 `.astype(np.float32)` 转换（~μs 级）

**注意**：指纹值不是二值的（范围 -170..1080），不能用 uint8（会截断负值和 >255 的值，导致神经网络预测错误）

### 2. SharedMemory 共享 filter 权重

**文件**：`simpretro.py` + `shared_state.py`

**改动**：
- Net_orig 所有权重拼接为连续 float32 buffer（~64 MB）
- 放入 SharedMemory
- 子进程通过 `torch.from_numpy()` 零拷贝恢复权重

**收益**：filter 权重跨进程真正共享，不退化

**代价**：初始化时多一次权重拼接（~ms 级）

### 3. 释放 fingerprint_base dict

**文件**：`simpretro.py`

**改动**：构建完 template_fps 后，清空 `fingerprint_base` 模块级 dict 并 `gc.collect()`

**收益**：释放 ~2.4 GB 堆内存（与 template_fps 完全重复）

**代价**：无

### 4. rdchiral Reaction::run() 副本逻辑

**文件**：`rdchiral_cpp/rdchiral/reaction.cpp` + `rdchiral.hpp`

**改动**：
- `run()` 方法声明加 `const`
- `run()` 内部对 `m_template_r`/`m_template_p` 做 ROMol 浅拷贝
- 所有 `setAtomMapNum` 写入在副本上执行，原始模板不被修改
- 使用副本的 atom map（`rt_map_copy`/`pt_map_copy`）替代原始 map

**收益**：搜索时不会写入原始模板页面，避免搜索导致的 COW 退化

**代价**：每次 `run()` 多一次 ROMol 拷贝（~1-10 μs/模板）

### 5. rdchiral 构造函数只读化

**文件**：`rdchiral_cpp/rdchiral/reaction.cpp` + `bonds.cpp`

**改动**：
- 跳过 `template_atom_could_have_been_tetra()` 预计算（避免写入原子 `"tetra_possible"` 属性）
- 删除 `enumerate_possible_cistrans_defs()` 中的 `setProp("localChirality")` 写入（该属性从未被读取）

**收益**：减少构造函数对模板对象的写入

**代价**：`tetra_possible` 在 `run()` 中按需计算（在副本上）

### 6. filter.eval()

**文件**：`simpretro.py`

**改动**：加载 filter 后调用 `self.filter.eval()`

**收益**：防止 BatchNorm 在推理时写入 running_mean/running_var

**代价**：无

### 7. valid_template_id O(1) 查找

**文件**：`simpretro.py`

**改动**：`list.index()` O(N) 线性搜索 → 预构建 dict O(1) 查找

**收益**：匹配结果多时过滤速度提升

**代价**：多构建一个 dict（~μs 级）

## 文件清单

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `shared_state.py` | 新建 | SharedArray + SharedWeights 封装 |
| `simpretro.py` | 修改 | SharedMemory、dtype 压缩、释放 dict、filter.eval()、O(1) 查找 |
| `rdchiral/reaction.cpp` | 修改 | run() 副本逻辑、跳过 tetra_possible |
| `rdchiral/rdchiral.hpp` | 修改 | run() 声明加 const |
| `rdchiral/bonds.cpp` | 修改 | 删除 localChirality 写入 |

## 效果

| 指标 | 改前 | 改后 |
|---|---|---|
| RSS per process | ~13 GB | ~11.4 GB |
| PSS per process | ~4.3 GB | ~3.8 GB |
| 总物理内存 | ~13 GB | ~12 GB |
| 搜索时 COW 退化 | 有 | 无 |

## 注意事项

1. **`ps` 报告的 RSS 不是真实占用**：RSS 含共享页重复计数。真实占用看 `PSS` 或 `free -h`。

2. **SharedMemory 段清理**：`create=True` 时会自动清理上次残留的 `/dev/shm/simpretro_*` 段。主进程 atexit 时 unlink。

3. **fork 前的脏页无法消除**：C++ TemplateLibrary 和 Python 堆分配在 fork 前就已完成，这些页面的 Shared_Dirty 是 fork 模式的固有成本。

4. **只支持 fork 模式**：spawn 模式下子进程不继承父进程内存，SharedMemory 需要额外 attach 逻辑，且 C++ TemplateLibrary 无法共享。
