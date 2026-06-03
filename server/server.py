import asyncio
import hashlib
import logging
import multiprocessing as mp
import os
import pickle
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# 必须在导入任何库之前设置：关闭 OpenMP/MKL 线程池，避免 fork 时子进程死锁。
# NumPy、PyTorch、RDKit 等库在首次导入时会初始化 OpenMP 线程池，
# fork 只复制当前线程，子进程中 OpenMP barrier 等待的线程不存在，永久阻塞。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from fastapi import FastAPI, HTTPException

try:
    # 被 uvicorn 从父目录导入时
    from server.schemas import SearchRequest, SearchResponse, TaskSubmitResponse, TaskStatusResponse
    from server.worker import init_worker, run_search
except ImportError:
    # 直接运行 server.py 时
    from schemas import SearchRequest, SearchResponse, TaskSubmitResponse, TaskStatusResponse
    from worker import init_worker, run_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 环境变量设置
# ============================================================================

class Settings:
    """对齐 search.py 默认加载参数。"""

    INVENTORY_PATH: str = os.getenv(
        "INVENTORY_PATH",
        "/home/liwenlong/chemTools/retro_syn/syntheseus/emolecules.txt",
    )
    CANONICALIZE_INVENTORY: bool = os.getenv(
        "CANONICALIZE_INVENTORY", "false"
    ).lower() in ("true", "1", "yes")

    MODEL_DIR: str = os.getenv(
        "MODEL_DIR",
        "/home/liwenlong/chemTools/retro_syn/syntheseus/syntheseus/SimpRetro_templates copy.json",
    )
    USE_GPU: bool = os.getenv("USE_GPU", "true").lower() in ("true", "1", "yes")
    REACTION_MODEL_USE_CACHE: bool = True
    NUM_TOP_RESULTS: int = int(os.getenv("NUM_TOP_RESULTS", "50"))

    # Process pool settings
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "2"))

    # Graph storage
    GRAPH_DIR: str = os.getenv("GRAPH_DIR", "./search_graphs")


settings = Settings()

# 创建 graph 存储目录
Path(settings.GRAPH_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================================
# 请求结果缓存（针对重复请求，减少搜索开销）
# ============================================================================

class ResponseCache:
    """LRU 缓存，键为请求参数的 SHA256，仅由容量上限控制淘汰。"""

    def __init__(self, max_size: int = 100):
        self._max_size = max_size
        self._store: dict[str, object] = {}
        self._order: list[str] = []  # 访问顺序，最早在末尾
        self._lock = threading.Lock()

    def _make_key(self, smiles: str, backend: str, options: dict) -> str:
        raw = f"{smiles}|{backend}|{options}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, smiles: str, backend: str, options: dict) -> Optional[object]:
        key = self._make_key(smiles, backend, options)
        with self._lock:
            if key not in self._store:
                return None
            # 命中：移到末尾（最近使用）
            self._order.remove(key)
            self._order.append(key)
            return self._store[key]

    def put(self, smiles: str, backend: str, options: dict, value: object) -> None:
        key = self._make_key(smiles, backend, options)
        with self._lock:
            if key in self._store:
                self._order.remove(key)
            elif len(self._store) >= self._max_size:
                old = self._order.pop(0)
                del self._store[old]
            self._store[key] = value
            self._order.append(key)


# 缓存：最多 100 条，永不过期（峰值约 2~20MB）
_cache = ResponseCache(max_size=100)


# ============================================================================
# 异步任务存储
# ============================================================================

class TaskStore:
    """线程安全的异步任务存储，支持过期清理。"""

    def __init__(self, ttl_seconds: int = 3600):
        self._ttl = ttl_seconds
        self._tasks: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        task_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._tasks[task_id] = {
                "status": "pending",
                "result": None,
                "error": None,
                "created_at": time.time(),
            }
        return task_id

    def set_running(self, task_id: str) -> None:
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = "running"

    def set_completed(self, task_id: str, result: dict) -> None:
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = "completed"
                self._tasks[task_id]["result"] = result

    def set_failed(self, task_id: str, error: str) -> None:
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = "failed"
                self._tasks[task_id]["error"] = error

    def get(self, task_id: str) -> Optional[dict]:
        with self._lock:
            self._cleanup_expired()
            return self._tasks.get(task_id)

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [tid for tid, t in self._tasks.items() if now - t["created_at"] > self._ttl]
        for tid in expired:
            # 清理关联的 graph 文件
            graph_path = Path(settings.GRAPH_DIR) / f"{tid}.pkl"
            if graph_path.exists():
                graph_path.unlink()
            del self._tasks[tid]


_task_store = TaskStore(ttl_seconds=3600)


# ============================================================================
# 进程池创建（在模块导入时执行，此时 uvicorn 事件循环未启动，fork 安全）
# ============================================================================

_pool: ProcessPoolExecutor = None


def _create_pool() -> ProcessPoolExecutor:
    """创建进程池。CPU 模式用 fork，GPU 模式用 spawn。"""
    max_workers = settings.MAX_WORKERS
    use_gpu = settings.USE_GPU

    if use_gpu:
        logger.info("Starting process pool with %d GPU worker(s) (spawn mode) ...", max_workers)
        ctx = mp.get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=init_worker,
            initargs=(set(), settings.CANONICALIZE_INVENTORY),
        )
    else:
        # CPU 模式：主进程预编译模板，fork 后 worker 通过 copy-on-write 共享内存
        logger.info("Pre-compiling templates in main process (fork mode) ...")
        inventory_smiles = set()
        init_worker(inventory_smiles, settings.CANONICALIZE_INVENTORY)
        logger.info("Templates pre-compiled, inventory ready in main process")

        ctx = mp.get_context("fork")
        pool = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
        )

    # 预 fork worker 进程（模块导入时是单线程，fork 安全）
    logger.info("Pre-forking %d worker processes ...", max_workers)
    import time
    futures = [pool.submit(time.sleep, 0) for _ in range(max_workers)]
    for f in futures:
        f.result()
    logger.info("预热加载完成：进程池已就绪 (%d workers)", max_workers)
    return pool


_pool = _create_pool()


# ============================================================================
# FastAPI 应用
# ============================================================================

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Retro Synthesis API started")
    yield
    logger.info("Shutting down process pool ...")
    _pool.shutdown(wait=False)
    logger.info("Process pool shut down")


app = FastAPI(
    title="Retro Synthesis Search API",
    description="逆合成规划搜索服务，预热加载模型和库存",
    version="1.0.0",
    lifespan=lifespan,
)


def _process_graph_result(result_dict: dict, task_id: str) -> dict:
    """保存 graph 并设置 graph_id，从结果中移除 graph 对象。"""
    graph_obj = result_dict.pop("graph", None)
    if graph_obj is not None:
        graph_path = Path(settings.GRAPH_DIR) / f"{task_id}.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(graph_obj, f)
        logger.info("Saved graph to %s", graph_path)
        result_dict["graph_id"] = task_id
    return result_dict


async def _run_search_task(task_id: str, req: SearchRequest, cache_key_options: dict):
    """后台执行搜索任务。"""
    _task_store.set_running(task_id)
    executor_timeout = req.build_tree_options.expansion_time + 60

    def _run_with_timeout() -> dict:
        future = _pool.submit(run_search, req)
        try:
            return future.result(timeout=executor_timeout)
        except Exception:
            future.cancel()
            raise

    try:
        result_dict = await asyncio.to_thread(_run_with_timeout)
        result_dict = _process_graph_result(result_dict, task_id)
        _cache.put(req.smiles, req.backend, cache_key_options, result_dict)
        _task_store.set_completed(task_id, result_dict)
        logger.info("Task %s completed", task_id)
    except TimeoutError:
        _task_store.set_failed(task_id, f"Search timed out after {req.build_tree_options.expansion_time}s")
    except ValueError as e:
        _task_store.set_failed(task_id, str(e))
    except Exception as e:
        logger.exception("Task %s failed", task_id)
        _task_store.set_failed(task_id, f"Search failed: {e}")


@app.post("/api/retrosynthesis/search", response_model=SearchResponse)
async def retrosynthesis_search(req: SearchRequest):
    """阻塞式搜索：等待结果完成后返回。"""
    if not req.smiles.strip():
        raise HTTPException(status_code=400, detail="SMILES is required")

    # 先查缓存
    options_dict = req.build_tree_options.model_dump()
    cached = _cache.get(req.smiles, req.backend, options_dict)
    if cached is not None:
        logger.info("Cache hit for smiles=%s backend=%s", req.smiles[:12], req.backend)
        return SearchResponse(**cached)

    # Executor 超时 = 搜索时间上限 + 60s 兜底
    executor_timeout = req.build_tree_options.expansion_time + 60

    def _run_with_timeout() -> dict:
        future = _pool.submit(run_search, req)
        try:
            return future.result(timeout=executor_timeout)
        except Exception:
            future.cancel()
            raise

    try:
        result_dict = await asyncio.to_thread(_run_with_timeout)
        graph_id = uuid.uuid4().hex[:12]
        result_dict = _process_graph_result(result_dict, graph_id)
        _cache.put(req.smiles, req.backend, options_dict, result_dict)
        return SearchResponse(**result_dict)
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Search timed out after {req.build_tree_options.expansion_time}s",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@app.post("/api/retrosynthesis/search/async", response_model=TaskSubmitResponse)
async def retrosynthesis_search_async(req: SearchRequest):
    """异步搜索：立即返回 task_id，通过轮询获取结果。"""
    if not req.smiles.strip():
        raise HTTPException(status_code=400, detail="SMILES is required")

    # 先查缓存
    options_dict = req.build_tree_options.model_dump()
    cached = _cache.get(req.smiles, req.backend, options_dict)
    if cached is not None:
        logger.info("Cache hit for smiles=%s backend=%s", req.smiles[:12], req.backend)
        task_id = _task_store.create()
        _task_store.set_completed(task_id, cached)
        return TaskSubmitResponse(task_id=task_id, status="completed")

    # 创建异步任务
    task_id = _task_store.create()
    logger.info("Task %s created for smiles=%s backend=%s", task_id, req.smiles[:12], req.backend)

    asyncio.create_task(_run_search_task(task_id, req, options_dict))
    return TaskSubmitResponse(task_id=task_id, status="pending")


@app.get("/api/retrosynthesis/status/{task_id}", response_model=TaskStatusResponse)
async def task_status(task_id: str):
    """查询异步任务状态。"""
    task = _task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found or expired")

    result = None
    if task["result"] is not None:
        result = SearchResponse(**task["result"])

    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        result=result,
        error=task["error"],
    )


@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.server:app", host="0.0.0.0", port=5000)
