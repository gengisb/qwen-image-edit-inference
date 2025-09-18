import os, errno

def _first_writable(paths):
    for p in paths:
        try:
            os.makedirs(p, exist_ok=True)
            test = os.path.join(p, ".write_test")
            open(test, "w").close(); os.remove(test)
            return p
        except Exception:
            continue
    raise OSError(errno.EROFS, "No writable offload directory found")

OFFLOAD_DIR = _first_writable([
    "/opt/ml/tmp/offload",   # EBS-backed and writable
    "/tmp/offload",          # always writable
])

HF_CACHE_DIR = _first_writable([
    "/opt/ml/tmp/hf-cache",
    "/tmp/hf-cache",
])

os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)
os.environ.setdefault("TORCH_HOME", "/opt/ml/tmp/torch" if os.path.isdir("/opt/ml/tmp") else "/tmp/torch")
