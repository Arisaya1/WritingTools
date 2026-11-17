"""
Minimal GGUF header reader.

Reads just enough of a .gguf file's metadata (architecture, quantization, trained
context length, layer count...) to show helpful info in the llama.cpp settings UI.
It never loads tensors and stops at the tokenizer section, so it is fast even on
multi-gigabyte files.
"""

import os
import struct

# GGUF metadata value types -> (struct format, size in bytes)
_SCALARS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
    6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
_STRING, _ARRAY = 8, 9
_MAX_STRING = 1 << 20  # Anything bigger is not a plain metadata string; bail out.
_MAX_KV = 20000

# llama_ftype values from llama.h (general.file_type).
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
}


class _Reader:
    def __init__(self, f):
        self.f = f

    def unpack(self, fmt: str, size: int):
        data = self.f.read(size)
        if len(data) != size:
            raise EOFError
        return struct.unpack(fmt, data)[0]

    def string(self) -> str:
        n = self.unpack("<Q", 8)
        if n > _MAX_STRING:
            raise ValueError("string too large")
        data = self.f.read(n)
        if len(data) != n:
            raise EOFError
        return data.decode("utf-8", errors="replace")

    def skip_string(self):
        n = self.unpack("<Q", 8)
        self.f.seek(n, os.SEEK_CUR)

    def value(self, vtype: int, keep: bool = True):
        if vtype in _SCALARS:
            fmt, size = _SCALARS[vtype]
            return self.unpack(fmt, size)
        if vtype == _STRING:
            if keep:
                return self.string()
            self.skip_string()
            return None
        if vtype == _ARRAY:
            etype = self.unpack("<I", 4)
            count = self.unpack("<Q", 8)
            if etype in _SCALARS:
                self.f.seek(count * _SCALARS[etype][1], os.SEEK_CUR)
            elif etype == _STRING:
                for _ in range(count):
                    self.skip_string()
            else:
                raise ValueError("unsupported array element type")
            return None
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf_info(path: str) -> dict | None:
    """
    Return a dict describing the model, or None if the file is not a readable GGUF.

    Keys: size_bytes, architecture, name, size_label, quantization, context_length,
    layers (any may be missing).
    """
    try:
        size_bytes = os.path.getsize(path)
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            r = _Reader(f)
            version = r.unpack("<I", 4)
            if version < 2:
                return {"size_bytes": size_bytes}
            r.unpack("<Q", 8)  # tensor count
            kv_count = min(r.unpack("<Q", 8), _MAX_KV)
            meta = {}
            for _ in range(kv_count):
                key = r.string()
                vtype = r.unpack("<I", 4)
                # The tokenizer section is huge and comes after everything we need.
                if key.startswith("tokenizer."):
                    break
                value = r.value(vtype)
                if value is not None:
                    meta[key] = value
    except (OSError, EOFError, ValueError, struct.error):
        return None

    info = {"size_bytes": size_bytes}
    arch = meta.get("general.architecture")
    if arch:
        info["architecture"] = arch
    if meta.get("general.name"):
        info["name"] = meta["general.name"]
    if meta.get("general.size_label"):
        info["size_label"] = meta["general.size_label"]
    if "general.file_type" in meta:
        info["quantization"] = FILE_TYPES.get(meta["general.file_type"], f"type {meta['general.file_type']}")
    if arch:
        if f"{arch}.context_length" in meta:
            info["context_length"] = int(meta[f"{arch}.context_length"])
        if f"{arch}.block_count" in meta:
            info["layers"] = int(meta[f"{arch}.block_count"])
    return info


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def describe(info: dict | None) -> str:
    """One-line human-readable summary for the UI."""
    if info is None:
        return "Not a valid GGUF file"
    parts = []
    if info.get("architecture"):
        parts.append(info["architecture"])
    if info.get("size_label"):
        parts.append(info["size_label"])
    if info.get("quantization"):
        parts.append(info["quantization"])
    parts.append(format_size(info["size_bytes"]))
    if info.get("layers"):
        parts.append(f"{info['layers']} layers")
    if info.get("context_length"):
        parts.append(f"trained context {info['context_length']:,}")
    return " · ".join(parts)


def find_models(folder: str, max_depth: int = 4, limit: int = 500) -> list:
    """
    Recursively find .gguf model files under `folder`. Skips multimodal projector
    files (mmproj*) and non-first shards of split models (llama.cpp loads those
    automatically from the first shard).
    """
    found = []
    base_depth = folder.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(folder):
        if root.count(os.sep) - base_depth >= max_depth:
            dirs[:] = []
        for name in sorted(files):
            lower = name.lower()
            if not lower.endswith(".gguf") or lower.startswith("mmproj"):
                continue
            if "-of-" in lower and "-00001-of-" not in lower:
                continue
            found.append(os.path.join(root, name))
            if len(found) >= limit:
                return found
    return found
