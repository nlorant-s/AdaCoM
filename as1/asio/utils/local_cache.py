"""Cache shared-filesystem model files to node-local /tmp/ to avoid
CPFS/NFS concurrent-read corruption (e.g., tokenizer 'trailing characters' errors)."""

import hashlib
import logging
import os
import shutil

logger = logging.getLogger(__name__)

_CACHE_ROOT = "/tmp/tokenizer_cache"


def ensure_local_tokenizer(model_path: str) -> str:
    """If *model_path* is a directory on a shared filesystem, copy it to
    node-local ``/tmp/`` and return the local path.  Uses a filelock so
    only one process per node does the copy.

    If *model_path* is not a local directory (e.g. a HuggingFace repo ID),
    it is returned unchanged.
    """
    if not model_path or not os.path.isdir(model_path):
        return model_path

    path_hash = hashlib.md5(model_path.encode()).hexdigest()[:8]
    local_dir = os.path.join(
        _CACHE_ROOT,
        f"{os.path.basename(model_path)}_{path_hash}",
    )
    marker = os.path.join(local_dir, "tokenizer_config.json")

    if os.path.exists(marker):
        return local_dir

    try:
        import filelock
    except ImportError:
        logger.warning("filelock not installed, skipping local cache")
        return model_path

    os.makedirs(_CACHE_ROOT, exist_ok=True)
    lock = filelock.FileLock(f"{local_dir}.lock", timeout=120)
    with lock:
        if not os.path.exists(marker):
            logger.info("Caching tokenizer %s -> %s", model_path, local_dir)
            shutil.copytree(model_path, local_dir, dirs_exist_ok=True)

    return local_dir
