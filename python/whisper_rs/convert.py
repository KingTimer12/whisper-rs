"""Wrapper around the official ct2-transformers-converter.

Model conversion is weight I/O and tensor renaming, runs once per model, and the
official converter tracks CTranslate2 format changes -- so it stays in Python
rather than being reimplemented in Rust.

Requires the optional dependencies: pip install "whisper-rs[convert]"
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["convert_model"]

_MISSING = (
    "ct2-transformers-converter was not found. Install the conversion extras:\n"
    '    pip install "whisper-rs[convert]"\n'
    "or directly:\n"
    "    pip install ctranslate2 transformers"
)


def convert_model(
    model: str,
    output_dir: str | Path,
    *,
    quantization: str | None = None,
    force: bool = False,
) -> str:
    """Convert a Hugging Face Whisper checkpoint to CTranslate2 format.

    Args:
        model: HF model id or local path, e.g. "openai/whisper-large-v3".
        output_dir: Directory to write the converted model into.
        quantization: Optional target type, e.g. "float16" or "int8".
        force: Overwrite output_dir if it already exists.

    Returns:
        The output directory as a string, ready to pass to WhisperModel.

    Raises:
        RuntimeError: The converter is not installed, or it failed.
    """
    if shutil.which("ct2-transformers-converter") is None:
        raise RuntimeError(_MISSING)

    output = Path(output_dir)
    if output.exists() and not force:
        raise RuntimeError(f"{output} already exists; pass force=True to overwrite")

    cmd = [
        "ct2-transformers-converter",
        "--model",
        str(model),
        "--output_dir",
        str(output),
    ]
    if quantization:
        cmd += ["--quantization", quantization]
    if force:
        cmd.append("--force")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ct2-transformers-converter failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )

    return str(output)


def _main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m whisper_rs.convert",
        description="Convert a Whisper checkpoint to CTranslate2 format.",
    )
    parser.add_argument("model", help='HF model id or local path, e.g. "openai/whisper-large-v3"')
    parser.add_argument("output_dir", help="Directory to write the converted model into")
    parser.add_argument("--quantization", default=None, help='e.g. "float16" or "int8"')
    parser.add_argument("--force", action="store_true", help="Overwrite output_dir")
    args = parser.parse_args(argv)

    try:
        path = convert_model(
            args.model,
            args.output_dir,
            quantization=args.quantization,
            force=args.force,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
