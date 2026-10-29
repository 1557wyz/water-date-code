"""Isolated experiment paths, strict JSON and immutable-file receipts."""
from __future__ import annotations
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
WORKSPACE = REPO.parent
CORE = REPO / "water_plant_layout_v3.py"


def read_bytes(path):
    path = Path(path)
    if os.name != "nt":
        return path.read_bytes()
    import msvcrt
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.restype = ctypes.c_void_p
    handle = kernel.CreateFileW(str(path), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_last_error(), str(path))
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    with os.fdopen(fd, "rb") as stream:
        return stream.read()


def digest(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def load(path):
    return json.loads(read_bytes(path).decode("utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def protected_files():
    excluded = {".git", "__pycache__", ".pytest_cache", ".revision_dependencies"}
    for directory, subdirs, names in os.walk(WORKSPACE):
        subdirs[:] = [name for name in subdirs if name not in excluded
                      and Path(directory, name) != ROOT]
        for name in names:
            path = Path(directory, name)
            if not name.startswith("~$") and path.suffix not in (".pyc", ".nbc", ".nbi"):
                yield path
    manuscript = WORKSPACE.parent / "manuscript.docx"
    if manuscript.exists():
        yield manuscript


def snapshot():
    path = ROOT / "audit/original_files_before.json"
    if path.exists():
        raise RuntimeError("baseline receipt already exists; refusing to replace it")
    files = {}
    for item in protected_files():
        files[str(item)] = {"sha256": digest(item), "bytes": item.stat().st_size}
    save(path, {"files": files, "excluded": ["new independent experiment directory", ".git",
        "dependency cache", "interpreter caches", "Office lock files"]})
    print(f"Protected {len(files)} pre-existing files")


def check_unchanged():
    before = load(ROOT / "audit/original_files_before.json")["files"]
    differences = []
    for filename, item in before.items():
        p = Path(filename)
        if not p.exists() or digest(p) != item["sha256"]:
            differences.append(filename)
    added = sorted(str(p) for p in protected_files() if str(p) not in before)
    result = {"checked_files": len(before), "changed_or_missing": differences,
              "new_files_outside_isolation": added, "all_original_files_unchanged": not differences,
              "core_sha256": digest(CORE)}
    save(ROOT / "audit/original_files_after_check.json", result)
    return result


if __name__ == "__main__":
    if sys.argv[-1] == "snapshot":
        snapshot()
    elif sys.argv[-1] == "check":
        print(json.dumps(check_unchanged(), ensure_ascii=False))
