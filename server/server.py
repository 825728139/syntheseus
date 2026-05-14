import asyncio
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager

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
    from server.schemas import SearchRequest, SearchResponse
    from server.worker import init_worker, run_search
except ImportError:
    # 直接运行 server.py 时
    from schemas import SearchRequest, SearchResponse
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


settings = Settings()


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


@app.post("/api/retrosynthesis/search", response_model=SearchResponse)
async def retrosynthesis_search(req: SearchRequest):
    if not req.smiles.strip():
        raise HTTPException(status_code=400, detail="SMILES is required")

    # Executor 超时略大于请求超时，作为硬兜底
    executor_timeout = req.timeout + 10

    def _run_with_timeout() -> dict:
        future = _pool.submit(run_search, req)
        try:
            return future.result(timeout=executor_timeout)
        except Exception:
            future.cancel()
            raise

    try:
        result_dict = await asyncio.to_thread(_run_with_timeout)
        return SearchResponse(**result_dict)
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Search timed out after {req.timeout}s",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.server:app", host="0.0.0.0", port=5000)
