from typing import TYPE_CHECKING

from joblib import Parallel
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from typing import Iterable


def parallel_run(tasks: "Iterable", workers: int, jobs: int = None):
    executor = Parallel(n_jobs=workers, return_as="generator")
    with tqdm(total=jobs) as pbar:
        results = []
        for result in executor(tasks):
            results.append(result)
            pbar.update(1)  # 只有当一个任务真正返回结果时，进度条才会跳动
    return results
