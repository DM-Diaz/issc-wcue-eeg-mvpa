#!/usr/bin/env python3
"""
collect_system_info.py

Collect reproducibility-oriented system and software metadata without recording
usernames, hostnames, MAC addresses, serial numbers, or full filesystem paths.

Outputs:
  - system_info_<timestamp>.json
  - system_info_<timestamp>.txt
  - pip_freeze_<timestamp>.txt   (optional; enabled with --full-freeze)

Recommended:
  Run this script inside the exact Python environment used for analysis.

Examples:
  python collect_system_info.py
  python collect_system_info.py --output-dir reproducibility --full-freeze
  python collect_system_info.py --note "MVPA model environment"
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SELECTED_PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "mne",
    "joblib",
    "matplotlib",
    "torch",
    "torchvision",
    "torchaudio",
    "statsmodels",
    "xgboost",
    "lightgbm",
    "numba",
    "h5py",
    "openpyxl",
]

THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "PYTHONHASHSEED",
]


def run_command(command: list[str], timeout: int = 20) -> dict[str, Any]:
    """Run a command safely and return a structured result."""
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "available": False,
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"{command[0]} not found",
        }

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "available": True,
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {
            "available": True,
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
        }


def get_cpu_model() -> str | None:
    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass

    if system == "Darwin":
        result = run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if result["returncode"] == 0 and result["stdout"]:
            return result["stdout"]

    if system == "Windows":
        result = run_command(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"]
        )
        if result["returncode"] == 0 and result["stdout"]:
            return result["stdout"]

    return platform.processor() or None


def get_memory_bytes() -> int | None:
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().total)
    except Exception:
        pass

    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kib = int(line.split()[1])
                        return kib * 1024
        except OSError:
            pass

    if platform.system() == "Darwin":
        result = run_command(["sysctl", "-n", "hw.memsize"])
        if result["returncode"] == 0:
            try:
                return int(result["stdout"])
            except ValueError:
                pass

    if platform.system() == "Windows":
        result = run_command(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"]
        )
        if result["returncode"] == 0:
            try:
                return int(result["stdout"])
            except ValueError:
                pass

    return None


def bytes_to_gib(value: int | None) -> float | None:
    return None if value is None else round(value / (1024 ** 3), 2)


def get_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in SELECTED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def get_numpy_config() -> str | None:
    try:
        np = importlib.import_module("numpy")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            np.show_config()
        text = buffer.getvalue().strip()
        return text or None
    except Exception:
        return None


def get_torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {"installed": False}
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return info

    info["installed"] = True
    info["version"] = getattr(torch, "__version__", None)
    info["cuda_build_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    info["cudnn_version"] = None
    info["cuda_available"] = False
    info["gpu_count"] = 0
    info["gpus"] = []

    try:
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        pass

    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu_count"] = int(torch.cuda.device_count())
        for idx in range(info["gpu_count"]):
            props = torch.cuda.get_device_properties(idx)
            info["gpus"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gib": round(props.total_memory / (1024 ** 3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    except Exception as exc:
        info["cuda_query_error"] = repr(exc)

    return info


def get_nvidia_smi_info() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    result = run_command(query)
    parsed: list[dict[str, Any]] = []

    if result["returncode"] == 0 and result["stdout"]:
        for line in result["stdout"].splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                parsed.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "driver_version": parts[2],
                        "memory_total_mib": parts[3],
                        "compute_capability": parts[4],
                    }
                )

    return {
        "query_result": result,
        "parsed_gpus": parsed,
    }


def get_git_info() -> dict[str, Any]:
    inside = run_command(["git", "rev-parse", "--is-inside-work-tree"])
    if inside["returncode"] != 0 or inside["stdout"].lower() != "true":
        return {"inside_repository": False}

    commit = run_command(["git", "rev-parse", "HEAD"])
    branch = run_command(["git", "branch", "--show-current"])
    status = run_command(["git", "status", "--porcelain"])
    describe = run_command(["git", "describe", "--always", "--dirty", "--tags"])

    return {
        "inside_repository": True,
        "commit": commit["stdout"] or None,
        "branch": branch["stdout"] or None,
        "describe": describe["stdout"] or None,
        "dirty": bool(status["stdout"]),
    }


def get_python_info() -> dict[str, Any]:
    return {
        "version": sys.version.replace("\n", " "),
        "implementation": platform.python_implementation(),
        "executable_name": Path(sys.executable).name,
        "prefix_name": Path(sys.prefix).name,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "virtual_environment_name": (
            Path(os.environ["VIRTUAL_ENV"]).name
            if os.environ.get("VIRTUAL_ENV")
            else None
        ),
    }


def get_os_info() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python_architecture": platform.architecture()[0],
    }


def build_report(note: str | None) -> dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc).astimezone()
    memory_bytes = get_memory_bytes()

    return {
        "schema_version": "1.0",
        "collected_at_local": now.isoformat(),
        "collected_at_utc": now.astimezone(dt.timezone.utc).isoformat(),
        "note": note,
        "privacy": {
            "hostname_recorded": False,
            "username_recorded": False,
            "serial_numbers_recorded": False,
            "full_paths_recorded": False,
        },
        "operating_system": get_os_info(),
        "hardware": {
            "cpu_model": get_cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "physical_cpu_count": (
                __import__("psutil").cpu_count(logical=False)
                if importlib.util.find_spec("psutil")
                else None
            ),
            "total_memory_bytes": memory_bytes,
            "total_memory_gib": bytes_to_gib(memory_bytes),
            "nvidia_smi": get_nvidia_smi_info(),
        },
        "python": get_python_info(),
        "selected_package_versions": get_package_versions(),
        "torch": get_torch_info(),
        "numpy_build_configuration": get_numpy_config(),
        "threading_and_device_environment": {
            key: os.environ.get(key) for key in THREAD_ENV_VARS
        },
        "git": get_git_info(),
    }


def render_text(report: dict[str, Any]) -> str:
    lines: list[str] = []

    def add(label: str, value: Any) -> None:
        lines.append(f"{label}: {value}")

    lines.append("SYSTEM AND SOFTWARE INFORMATION")
    lines.append("=" * 32)
    add("Collected (local)", report["collected_at_local"])
    add("Collected (UTC)", report["collected_at_utc"])
    add("Note", report["note"])
    lines.append("")

    os_info = report["operating_system"]
    lines.append("OPERATING SYSTEM")
    lines.append("-" * 16)
    add("System", os_info["system"])
    add("Release", os_info["release"])
    add("Version", os_info["version"])
    add("Architecture", os_info["architecture"])
    add("Python architecture", os_info["python_architecture"])
    lines.append("")

    hw = report["hardware"]
    lines.append("HARDWARE")
    lines.append("-" * 8)
    add("CPU", hw["cpu_model"])
    add("Logical CPU count", hw["logical_cpu_count"])
    add("Physical CPU count", hw["physical_cpu_count"])
    add("RAM (GiB)", hw["total_memory_gib"])

    gpus = hw["nvidia_smi"]["parsed_gpus"]
    if gpus:
        for gpu in gpus:
            add(
                f"GPU {gpu['index']}",
                f"{gpu['name']} | driver {gpu['driver_version']} | "
                f"{gpu['memory_total_mib']} MiB | compute capability {gpu['compute_capability']}",
            )
    else:
        add("NVIDIA GPU", "Not detected by nvidia-smi")
    lines.append("")

    py = report["python"]
    lines.append("PYTHON")
    lines.append("-" * 6)
    add("Version", py["version"])
    add("Implementation", py["implementation"])
    add("Executable name", py["executable_name"])
    add("Environment", py["conda_environment"] or py["virtual_environment_name"])
    lines.append("")

    lines.append("SELECTED PACKAGE VERSIONS")
    lines.append("-" * 25)
    for package, version in report["selected_package_versions"].items():
        add(package, version if version is not None else "not installed")
    lines.append("")

    torch_info = report["torch"]
    lines.append("PYTORCH / CUDA")
    lines.append("-" * 14)
    add("PyTorch installed", torch_info.get("installed"))
    if torch_info.get("installed"):
        add("PyTorch version", torch_info.get("version"))
        add("CUDA build version", torch_info.get("cuda_build_version"))
        add("cuDNN version", torch_info.get("cudnn_version"))
        add("CUDA available", torch_info.get("cuda_available"))
        add("GPU count", torch_info.get("gpu_count"))
        for gpu in torch_info.get("gpus", []):
            add(
                f"Torch GPU {gpu['index']}",
                f"{gpu['name']} | {gpu['total_memory_gib']} GiB | "
                f"compute capability {gpu['compute_capability']}",
            )
    lines.append("")

    lines.append("THREADING / DEVICE ENVIRONMENT")
    lines.append("-" * 30)
    for key, value in report["threading_and_device_environment"].items():
        add(key, value)
    lines.append("")

    git = report["git"]
    lines.append("GIT")
    lines.append("-" * 3)
    add("Inside repository", git.get("inside_repository"))
    if git.get("inside_repository"):
        add("Commit", git.get("commit"))
        add("Branch", git.get("branch"))
        add("Describe", git.get("describe"))
        add("Dirty working tree", git.get("dirty"))
    lines.append("")

    lines.append("NUMPY BUILD CONFIGURATION")
    lines.append("-" * 25)
    lines.append(report["numpy_build_configuration"] or "Unavailable")
    lines.append("")

    lines.append("PRIVACY")
    lines.append("-" * 7)
    lines.append(
        "This report intentionally omits hostname, username, hardware serial numbers, "
        "MAC addresses, and full filesystem paths."
    )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect reproducibility-oriented system metadata."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("system_info"),
        help="Directory for output files (default: ./system_info)",
    )
    parser.add_argument(
        "--full-freeze",
        action="store_true",
        help="Also save the complete output of `python -m pip freeze`.",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Optional short description of this environment.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    report = build_report(args.note)
    json_path = args.output_dir / f"system_info_{stamp}.json"
    txt_path = args.output_dir / f"system_info_{stamp}.txt"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    txt_path.write_text(render_text(report), encoding="utf-8")

    print(render_text(report))
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved TXT:  {txt_path}")

    if args.full_freeze:
        freeze_result = run_command([sys.executable, "-m", "pip", "freeze"], timeout=60)
        freeze_path = args.output_dir / f"pip_freeze_{stamp}.txt"
        freeze_path.write_text(
            freeze_result["stdout"] + (
                f"\n\n# STDERR\n{freeze_result['stderr']}"
                if freeze_result["stderr"]
                else ""
            ),
            encoding="utf-8",
        )
        print(f"Saved pip freeze: {freeze_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
