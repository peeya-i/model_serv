import asyncio
import json
import os
import signal
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
AGENTS_DIR = os.path.join(PROJECT_ROOT, "agents")
LOGS_FILE = os.path.join(PROJECT_ROOT, "logs", "events.json")
MODEL_START_TIME_FILE = os.path.join(PROJECT_ROOT, "logs", "model_start_time.txt")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
PYTHON_EXEC = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

app = FastAPI(title="Private LLM & Local Agent Server Console")

# Mount static directory for offline assets like Chart.js
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory tracking for background processes launched via Web App
running_processes = {
    "model": {
        "process": None,
        "model_name": None,
        "port": 8000,
        "pid": None,
        "started_at": None,
        "last_exit_code": None,
        "last_error": None,
    },
    "agent": {
        "process": None,
        "agent_file": None,
        "port": 8001,
        "pid": None,
        "started_at": None,
        "last_exit_code": None,
        "last_error": None,
    },
    "agents": {},  # agent_name -> state dict
}


def check_model_server_health(port: int = 8000) -> dict[str, Any]:
    """Checks if the vLLM OpenAI-compatible REST server is actively healthy and serving requests."""
    url_health = f"http://127.0.0.1:{port}/health"
    url_models = f"http://127.0.0.1:{port}/v1/models"
    headers = {"Authorization": "Bearer your-internal-secure-gateway-token-xyz"}

    # 1. Fast check on /health endpoint
    try:
        resp = requests.get(url_health, timeout=0.8)
        if resp.status_code == 200:
            active_model = None
            try:
                m_resp = requests.get(url_models, headers=headers, timeout=0.8)
                if m_resp.status_code == 200:
                    data = m_resp.json()
                    if data.get("data") and len(data["data"]) > 0:
                        active_model = data["data"][0].get("id")
            except Exception:
                pass
            return {"healthy": True, "active_model": active_model}
    except Exception:
        pass

    # 2. Fallback check on /v1/models directly
    try:
        m_resp = requests.get(url_models, headers=headers, timeout=0.8)
        if m_resp.status_code == 200:
            data = m_resp.json()
            active_model = None
            if data.get("data") and len(data["data"]) > 0:
                active_model = data["data"][0].get("id")
            return {"healthy": True, "active_model": active_model}
    except Exception:
        pass

    return {"healthy": False, "active_model": None}


def check_agent_server_health(port: int = 8001) -> bool:
    """Checks if the FastAPI agent microservice is healthy and ready to accept requests."""
    try:
        resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.8)
        return resp.status_code == 200
    except Exception:
        return False


def kill_processes_on_port(port: int):
    """Finds and forcefully terminates any process listening on the specified TCP port."""
    if not port:
        return
    try:
        cmd = ["lsof", "-ti", f":{port}"]
        pids_str = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        if pids_str:
            pids = [int(p) for p in pids_str.split() if p.isdigit()]
            for pid in pids:
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGTERM)
                except Exception:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except Exception:
                        pass
            time.sleep(0.3)
            for pid in pids:
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except Exception:
                        pass
    except Exception:
        pass


def kill_lingering_vllm_processes():
    """Finds and forcefully terminates any orphaned vLLM engine core or worker processes."""
    current_pid = os.getpid()
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                pid = proc.info["pid"]
                if pid == current_pid:
                    continue
                name = (proc.info.get("name") or "").lower()
                cmdline = " ".join(proc.info.get("cmdline") or []).lower()
                if "vllm::enginecore" in name or "enginecore" in name or "vllm::enginecore" in cmdline:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass

    if platform.system() in ["Linux", "Darwin"]:
        try:
            subprocess.run(["pkill", "-9", "-f", "VLLM::EngineCore"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def extract_model_server_error(log_path: str) -> str:
    """Extracts the actionable root cause error from the model server log."""
    if not os.path.exists(log_path):
        return "Process exited unexpectedly. Check logs."
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            return "Process exited unexpectedly (empty log)."

        # Look backwards for explicit error root causes
        for line in reversed(lines):
            for pattern in ("ValueError:", "OutOfMemoryError:", "torch.cuda.OutOfMemoryError:", "CUDA error:", "Free memory on device", "FileNotFoundError:", "OSError:", "RuntimeError:"):
                if pattern in line and "See root cause above" not in line:
                    idx = line.find(pattern)
                    return line[idx:]

        # Fallback to the last informative line
        for line in reversed(lines):
            if "See root cause above" not in line and not line.startswith("===") and "Traceback" not in line:
                clean_line = line.split("] ", 1)[-1] if "] " in line else line
                return clean_line

        return lines[-1]
    except Exception as e:
        return f"Error reading log: {e}"


def stop_running_agent(port: int = 8001):
    """Stops any currently running server agent to enforce mutual exclusion (only 1 running at a time)."""
    global running_processes
    # 1. Check running_processes["agent"]
    active = running_processes.get("agent")
    if active and active.get("process"):
        proc = active["process"]
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=2)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass

    # 2. Check running_processes["agents"] for any lingering processes
    for fname, info in list(running_processes.get("agents", {}).items()):
        if info and info.get("process"):
            p = info["process"]
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    p.wait(timeout=2)
                except Exception:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except Exception:
                        pass
            running_processes["agents"][fname]["process"] = None

    # 3. Kill any process listening on the agent port
    kill_processes_on_port(port)
    if active and active.get("port") and active["port"] != port:
        kill_processes_on_port(active["port"])

    running_processes["agent"] = {
        "process": None,
        "agent_file": None,
        "port": port,
        "pid": None,
        "started_at": None,
        "last_exit_code": 0,
        "last_error": None,
    }


def get_gpu_info() -> dict[str, Any]:
    """Detects comprehensive GPU hardware specifications and real-time VRAM availability."""
    cpu_cores = os.cpu_count() or 1
    info = {
        "model": "NVIDIA GPU (or Compatible Accelerator)",
        "capabilities": "Compute Capability 7.5 (Turing) | CUDA 13.2 | FP16/FP32 Engine",
        "processors": f"30 Multiprocessors (~1920 CUDA Cores) | {cpu_cores} CPU Threads",
        "num_gpus": 1,
        "ram_available_mb": 5736,
        "ram_total_mb": 6144,
        "ram_used_mb": 408,
        "driver_version": "N/A",
        "cuda_version": "13.2",
    }

    # Query via nvidia-smi
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,memory.used,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
        raw_output = subprocess.check_output(cmd, text=True, timeout=3).strip()
        if raw_output:
            lines = raw_output.splitlines()
            info["num_gpus"] = len(lines)
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) >= 6:
                info["model"] = parts[0]
                info["ram_total_mb"] = int(parts[1])
                info["ram_available_mb"] = int(parts[2])
                info["ram_used_mb"] = int(parts[3])
                info["driver_version"] = parts[4]
                cc = parts[5]
                arch = (
                    "Turing" if "7.5" in cc
                    else "Ampere" if "8." in cc
                    else "Ada Lovelace" if "8.9" in cc
                    else "Hopper / Blackwell" if "9." in cc
                    else "NVIDIA CUDA"
                )
                info["capabilities"] = f"Compute Capability {cc} ({arch}) | FP16/FP32 Acceleration"
    except Exception:
        pass

    # Enrich with PyTorch CUDA attributes if available
    try:
        import torch
        if torch.cuda.is_available():
            if info["num_gpus"] == 0:
                info["num_gpus"] = torch.cuda.device_count()
            if "NVIDIA" not in info["model"] and info["model"] == "Unknown":
                info["model"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            sms = props.multi_processor_count
            cores = sms * 64
            info["processors"] = f"{sms} Streaming Multiprocessors (~{cores} CUDA Cores) | {cpu_cores} CPU Threads"
    except Exception:
        pass

    return info


MODEL_METADATA_CATALOG = {
    "Llama-3.2-3B-Instruct": {
        "display_name": "Meta Llama 3.2 3B Instruct",
        "parameters": "3.2B",
        "vram_required_mb": 5120,
        "vram_str": "~4.8 - 5.2 GB",
        "quantization": "BF16 / FP16",
        "tool_support": True,
        "category": "agent_tools",
        "applications": [
            "Autonomous Agent Tool Calling (Native JSON Schemas)",
            "Multi-Turn Context Reasoning & Instruction Following",
            "Local Private RAG (Retrieval-Augmented Generation)",
            "Conversational Assistant & Enterprise Chatbot"
        ],
        "default_len": 4096,
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "display_name": "Meta Llama 3.2 3B Instruct (Hugging Face)",
        "parameters": "3.2B",
        "vram_required_mb": 5120,
        "vram_str": "~4.8 - 5.2 GB",
        "quantization": "BF16 / FP16",
        "tool_support": True,
        "category": "agent_tools",
        "applications": [
            "Autonomous Agent Tool Calling (Native JSON Schemas)",
            "Multi-Turn Context Reasoning & Instruction Following",
            "Local Private RAG (Retrieval-Augmented Generation)",
            "Conversational Assistant & Enterprise Chatbot"
        ],
        "default_len": 4096,
    },
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "display_name": "TinyLlama 1.1B Chat",
        "parameters": "1.1B",
        "vram_required_mb": 2200,
        "vram_str": "~2.0 - 2.5 GB",
        "quantization": "BF16 / FP16",
        "tool_support": False,
        "category": "fast_prototyping",
        "applications": [
            "Ultra-Fast Edge Prototyping & Diagnostics",
            "Resource-Constrained Environments (<4GB VRAM)",
            "High-Speed Low-Latency Message Completions",
            "API & Microservice Pipeline Health Verification"
        ],
        "default_len": 2048,
    },
    "Qwen/Qwen2.5-Coder-3B-Instruct": {
        "display_name": "Qwen 2.5 Coder 3B Instruct",
        "parameters": "3.1B",
        "vram_required_mb": 5200,
        "vram_str": "~5.0 - 5.5 GB",
        "quantization": "BF16 / FP16",
        "tool_support": True,
        "category": "coding",
        "applications": [
            "Multi-Language Code Generation (Python, JS, C++, Rust, Go)",
            "Syntax Debugging, Lint Repair, and Code Explanation",
            "Bash Shell Automation and Command Line Scripting",
            "Technical Tool Orchestration & API Client Generation"
        ],
        "default_len": 4096,
    },
    "google/gemma-2-2b-it": {
        "display_name": "Google Gemma 2 2B Instruct",
        "parameters": "2.6B",
        "vram_required_mb": 3800,
        "vram_str": "~3.6 - 4.0 GB",
        "quantization": "BF16 / FP16",
        "tool_support": False,
        "category": "qa_summarization",
        "applications": [
            "Factual Question Answering & Knowledge Synthesis",
            "Executive Article & Meeting Transcript Summarization",
            "Customer Support & Safe Content Dialogue",
            "Edge Computing Inference on Lower-Tier GPUs"
        ],
        "default_len": 4096,
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "display_name": "Meta Llama 3 8B Instruct",
        "parameters": "8.0B",
        "vram_required_mb": 8500,
        "vram_str": "~8.5 - 16.0 GB",
        "quantization": "FP16 / 4-bit AWQ",
        "tool_support": True,
        "category": "deep_reasoning",
        "applications": [
            "Deep Multi-Step Analytical Reasoning & Strategy",
            "Complex Long-Form Creative & Technical Writing",
            "High-Complexity Academic and Scientific Problem Solving",
            "Advanced Logical Planning (Requires 3GB+ CPU Offload on 6GB GPUs)"
        ],
        "default_len": 4096,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "display_name": "DeepSeek R1 Distill Qwen 1.5B",
        "parameters": "1.5B",
        "vram_required_mb": 2800,
        "vram_str": "~2.6 - 3.0 GB",
        "quantization": "BF16 / FP16",
        "tool_support": False,
        "category": "deep_reasoning",
        "applications": [
            "Chain-of-Thought (CoT) Mathematical Reasoning",
            "Algorithmic & Logical Puzzle Step-by-Step Solving",
            "Compact Coding & Mathematical Tutoring Assistants",
            "Fast Efficient Reasoning on 4GB-6GB GPUs"
        ],
        "default_len": 4096,
    },
    "microsoft/Phi-3.5-mini-instruct": {
        "display_name": "Microsoft Phi-3.5 Mini Instruct",
        "parameters": "3.8B",
        "vram_required_mb": 5800,
        "vram_str": "~5.5 - 6.2 GB",
        "quantization": "BF16 / FP16",
        "tool_support": False,
        "category": "qa_summarization",
        "applications": [
            "High-Quality Reasoning on Compact Mobile/Desktop GPUs",
            "Multi-Turn Multilingual Question Answering",
            "Structured Information Extraction & Document Auditing",
            "Context-Rich Analytical Summarization"
        ],
        "default_len": 4096,
    },
}

HF_SYSTEM_CACHE = os.path.expanduser("~/.cache/huggingface/hub")


def get_available_models_list(gpu_vram_mb: Optional[int] = None) -> list[dict[str, Any]]:
    """Scans local directories and cache, calculates hardware compatibility based on GPU VRAM,
    and enriches with target applications and specifications."""
    if gpu_vram_mb is None:
        try:
            gpu_info = get_gpu_info()
            gpu_vram_mb = gpu_info.get("ram_total_mb", 6144)
        except Exception:
            gpu_vram_mb = 6144

    active_model_name = running_processes.get("model", {}).get("model_name")
    models_dict: dict[str, dict[str, Any]] = {}

    # 1. Helper to record or update model entry
    def record_model(model_id: str, path_str: str, source_type: str, is_downloaded: bool = True):
        # Match against catalog metadata if available
        meta = MODEL_METADATA_CATALOG.get(model_id) or MODEL_METADATA_CATALOG.get(os.path.basename(model_id)) or {}
        display_name = meta.get("display_name") or model_id
        parameters = meta.get("parameters") or ("1.1B" if "1.1b" in model_id.lower() else "8B" if "8b" in model_id.lower() else "3B")
        vram_req = meta.get("vram_required_mb") or (2200 if "1.1b" in model_id.lower() else 8500 if "8b" in model_id.lower() else 5120)
        vram_str = meta.get("vram_str") or (f"~{round(vram_req/1024, 1)} GB")
        apps = meta.get("applications") or [
            "General Text Completion & Dialogue",
            "Instruction Following & Content Generation"
        ]
        category = meta.get("category") or "general"
        tool_support = meta.get("tool_support", ("llama-3.2" in model_id.lower() or "qwen" in model_id.lower()))
        default_len = meta.get("default_len") or (2048 if "tinyllama" in model_id.lower() else 4096)

        # Calculate GPU compatibility
        pct_gpu = round((vram_req / max(1, gpu_vram_mb)) * 100)
        if vram_req <= gpu_vram_mb:
            compat_status = "optimal"
            compat_badge = f"🟢 100% GPU Compatible ({pct_gpu}% VRAM)"
            compat_note = f"Fits comfortably inside {round(gpu_vram_mb/1024, 1)}GB GPU VRAM with dedicated KV cache"
        elif vram_req <= gpu_vram_mb + 4096:
            compat_status = "cpu_offload"
            needed_offload = round((vram_req - gpu_vram_mb) / 1024, 1)
            compat_badge = f"🟡 Runs with CPU Offload (~{needed_offload}GB RAM)"
            compat_note = f"Exceeds dedicated VRAM by ~{needed_offload}GB; automatically offloads weight layers to system RAM"
        else:
            compat_status = "insufficient"
            compat_badge = "🔴 Exceeds System Memory"
            compat_note = f"Requires ~{vram_str} which exceeds available hardware capacity"

        models_dict[model_id] = {
            "name": model_id,
            "display_name": display_name,
            "path": path_str,
            "parameters": parameters,
            "vram_required_mb": vram_req,
            "vram_str": vram_str,
            "vram_percent_of_gpu": pct_gpu,
            "compatibility_status": compat_status,
            "compatibility_badge": compat_badge,
            "compatibility_note": compat_note,
            "applications": apps,
            "category": category,
            "tool_support": tool_support,
            "max_model_len": default_len,
            "type": source_type,
            "is_downloaded": is_downloaded,
            "is_active": (active_model_name == model_id or active_model_name == path_str),
        }

    # 2. Scan models/ directory
    if os.path.isdir(MODELS_DIR):
        for entry in sorted(os.listdir(MODELS_DIR)):
            full_path = os.path.join(MODELS_DIR, entry)
            if entry.startswith("models--"):
                parts = entry.split("--")
                if len(parts) >= 3:
                    repo_id = f"{parts[1]}/{parts[2]}"
                    record_model(repo_id, repo_id, "Local HuggingFace Cache", is_downloaded=True)
            elif os.path.isdir(full_path) and not entry.startswith((".", "hub", "xet")):
                cfg_path = os.path.join(full_path, "config.json")
                if os.path.exists(cfg_path):
                    record_model(entry, f"models/{entry}", "Local Weights Directory", is_downloaded=True)

    # 3. Scan models/hub cache
    hub_dir = os.path.join(MODELS_DIR, "hub")
    if os.path.isdir(hub_dir):
        for entry in sorted(os.listdir(hub_dir)):
            if entry.startswith("models--"):
                parts = entry.split("--")
                if len(parts) >= 3:
                    repo_id = f"{parts[1]}/{parts[2]}"
                    if repo_id not in models_dict:
                        record_model(repo_id, repo_id, "Local Hub Cache", is_downloaded=True)

    # 4. Scan ~/.cache/huggingface/hub
    if os.path.isdir(HF_SYSTEM_CACHE):
        for entry in sorted(os.listdir(HF_SYSTEM_CACHE)):
            if entry.startswith("models--"):
                parts = entry.split("--")
                if len(parts) >= 3:
                    repo_id = f"{parts[1]}/{parts[2]}"
                    if repo_id not in models_dict:
                        record_model(repo_id, repo_id, "User HF Cache", is_downloaded=True)

    # 5. Include popular recommended models from catalog that can run on this system
    for model_id in MODEL_METADATA_CATALOG:
        if model_id not in models_dict and not any(m["name"].endswith(model_id) for m in models_dict.values()):
            record_model(model_id, model_id, "Hugging Face Model Hub", is_downloaded=False)

    # Return sorted list: Downloaded & Optimal models first, then other models
    sorted_models = sorted(
        models_dict.values(),
        key=lambda m: (
            not m["is_active"],
            not m["is_downloaded"],
            0 if m["compatibility_status"] == "optimal" else 1 if m["compatibility_status"] == "cpu_offload" else 2,
            m["vram_required_mb"],
        )
    )

    return sorted_models


def get_available_agents_list() -> list[dict[str, Any]]:
    """Scans the agents/ folder for server-side agent scripts."""
    agents = []
    seen_files = set()

    active_agent = running_processes.get("agent", {})
    active_proc = active_agent.get("process")
    active_file = active_agent.get("agent_file") if (active_proc and active_proc.poll() is None) else None

    if os.path.isdir(AGENTS_DIR):
        for fname in sorted(os.listdir(AGENTS_DIR)):
            if fname.endswith(".py") and not fname.startswith("__"):
                if fname in seen_files:
                    continue
                seen_files.add(fname)

                is_running = (fname == active_file)
                agent_port = active_agent.get("port", 8001) if is_running else 8001
                desc = "FastAPI HTTP Agent Microservice Front with automated tool reasoning"
                if "langchain" in fname:
                    desc = "LangChain Python Agent pipeline with sequential tool execution"

                agents.append({
                    "filename": fname,
                    "path": os.path.join("agents", fname),
                    "description": desc,
                    "default_port": agent_port,
                    "is_running": is_running,
                    "pid": active_agent.get("pid") if is_running else None,
                })
    return agents


# Pydantic Request Models
class StartModelRequest(BaseModel):
    model: str
    port: int = 8000


class StopModelRequest(BaseModel):
    port: Optional[int] = 8000


class StartAgentRequest(BaseModel):
    agent_file: str
    port: int = 8001
    model_port: int = 8000


class StopAgentRequest(BaseModel):
    agent_file: Optional[str] = None
    port: Optional[int] = 8001


class ChatQueryRequest(BaseModel):
    prompt: str
    target: str = "agent"  # "agent" or "model"
    port: int = 8001
    model: Optional[str] = None
    temperature: Optional[float] = 0.7


class TerminateAppRequest(BaseModel):
    shutdown_web: bool = False


# API Endpoints
@app.get("/api/system/gpu")
def api_gpu_info():
    return get_gpu_info()


@app.get("/api/models")
def api_models():
    return get_available_models_list()


@app.get("/api/agents")
def api_agents():
    return get_available_agents_list()


@app.get("/api/status")
def api_status():
    gpu = get_gpu_info()
    models = get_available_models_list()
    agents = get_available_agents_list()

    # --- Model Server Status Evaluation ---
    model_info = running_processes["model"]
    model_proc = model_info.get("process")
    model_port = model_info.get("port", 8000)
    model_state = "stopped"
    model_pid = None
    elapsed_sec = None
    error_msg = model_info.get("last_error")
    exit_code = model_info.get("last_exit_code")
    active_model_name = model_info.get("model_name")

    if model_proc:
        poll_res = model_proc.poll()
        if poll_res is None:
            model_pid = model_proc.pid
            started_at = model_info.get("started_at")
            if started_at:
                elapsed_sec = int(time.time() - started_at)

            # Test active readiness via health endpoint
            health = check_model_server_health(model_port)
            if health["healthy"]:
                model_state = "ready"
                if health["active_model"]:
                    active_model_name = health["active_model"]
                    model_info["model_name"] = health["active_model"]
            else:
                model_state = "loading"
        else:
            # Process terminated unexpectedly
            model_info["process"] = None
            model_info["last_exit_code"] = poll_res
            exit_code = poll_res
            log_path = os.path.join(PROJECT_ROOT, "logs", "model_server.log")
            error_msg = extract_model_server_error(log_path)
            model_info["last_error"] = error_msg
            model_state = "crashed" if poll_res != 0 else "stopped"
    else:
        # Check if an external or pre-existing model server is active on the port
        health = check_model_server_health(model_port)
        if health["healthy"]:
            model_state = "ready"
            if health["active_model"]:
                active_model_name = health["active_model"]
        else:
            if model_info.get("last_exit_code") not in (None, 0):
                model_state = "crashed"
            else:
                model_state = "stopped"

    # --- Agent Server Status Evaluation ---
    agent_info = running_processes["agent"]
    agent_proc = agent_info.get("process")
    agent_port = agent_info.get("port", 8001)
    agent_state = "stopped"
    agent_pid = None
    agent_elapsed = None
    active_agent_file = agent_info.get("agent_file")

    if agent_proc:
        poll_res = agent_proc.poll()
        if poll_res is None:
            agent_pid = agent_proc.pid
            started_at = agent_info.get("started_at")
            if started_at:
                agent_elapsed = int(time.time() - started_at)

            if check_agent_server_health(agent_port):
                agent_state = "ready"
            else:
                agent_state = "loading"
        else:
            agent_info["process"] = None
            agent_state = "crashed" if poll_res != 0 else "stopped"
    else:
        if check_agent_server_health(agent_port):
            agent_state = "ready"
            if not active_agent_file:
                active_agent_file = "agent_server.py"

    # Synchronize agents list is_running flags
    for ag in agents:
        if agent_state in ["loading", "ready"] and ag["filename"] == active_agent_file:
            ag["is_running"] = True
            ag["pid"] = agent_pid
            ag["status"] = agent_state
        else:
            ag["is_running"] = False
            ag["pid"] = None
            ag["status"] = "stopped"

    return {
        "gpu": gpu,
        "models": models,
        "agents": agents,
        "model_server": {
            "status": model_state,          # "stopped", "loading", "ready", "crashed"
            "running": (model_state == "ready"),
            "is_alive": (model_state in ["loading", "ready"]),
            "model_name": active_model_name,
            "port": model_port,
            "pid": model_pid,
            "elapsed_sec": elapsed_sec,
            "exit_code": exit_code,
            "last_error": error_msg,
        },
        "active_agent": {
            "status": agent_state,          # "stopped", "loading", "ready", "crashed"
            "running": (agent_state == "ready"),
            "is_alive": (agent_state in ["loading", "ready"]),
            "agent_file": active_agent_file,
            "port": agent_port,
            "pid": agent_pid,
            "elapsed_sec": agent_elapsed,
        }
    }


@app.post("/api/model/start")
def api_start_model(req: StartModelRequest):
    global running_processes
    # Check if already running or loading
    curr_proc = running_processes["model"]["process"]
    if curr_proc and curr_proc.poll() is None:
        return {
            "status": "already_running",
            "message": f"Model server is already running on port {running_processes['model']['port']} (PID: {curr_proc.pid})",
            "port": running_processes["model"]["port"],
        }

    # Clean up any lingering or orphaned worker processes from previous crashed sessions
    kill_lingering_vllm_processes()
    kill_processes_on_port(req.port)

    cmd = [
        PYTHON_EXEC,
        os.path.join(PROJECT_ROOT, "serve_private_llm.py"),
        "--cli",
        "--model", req.model,
        "--port", str(req.port),
    ]

    try:
        model_log_path = os.path.join(PROJECT_ROOT, "logs", "model_server.log")
        os.makedirs(os.path.dirname(model_log_path), exist_ok=True)
        with open(model_log_path, "w", encoding="utf-8") as f:
            f.write(f"=== Starting Model Server for {req.model} on port {req.port} ===\n")

        log_fd = open(model_log_path, "a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        log_fd.close()

        now_ts = time.time()
        running_processes["model"] = {
            "process": proc,
            "model_name": req.model,
            "port": req.port,
            "pid": proc.pid,
            "started_at": now_ts,
            "last_exit_code": None,
            "last_error": None,
        }
        try:
            with open(MODEL_START_TIME_FILE, "w", encoding="utf-8") as sf:
                sf.write(str(now_ts))
        except Exception:
            pass
        return {
            "status": "started",
            "message": f"Model server initializing for '{req.model}' on port {req.port}. Weights are loading into memory.",
            "pid": proc.pid,
            "port": req.port,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to launch model server: {e}")


@app.post("/api/model/stop")
def api_stop_model(req: Optional[StopModelRequest] = None):
    global running_processes
    port_to_stop = req.port if (req and req.port) else running_processes["model"].get("port", 8000)

    # 1. Terminate tracked model subprocess if running
    proc = running_processes["model"].get("process")
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=2)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass

    # 2. Terminate any processes listening on the port (handles standalone/CLI launches or lingering workers)
    kill_processes_on_port(port_to_stop)
    if running_processes["model"].get("port") and running_processes["model"]["port"] != port_to_stop:
        kill_processes_on_port(running_processes["model"]["port"])
    kill_lingering_vllm_processes()

    # 3. Wait up to 2 seconds for port to completely close
    for _ in range(15):
        if not check_model_server_health(port_to_stop)["healthy"]:
            break
        time.sleep(0.12)

    running_processes["model"] = {
        "process": None,
        "model_name": None,
        "port": port_to_stop,
        "pid": None,
        "started_at": None,
        "last_exit_code": 0,
        "last_error": None,
    }
    return {"status": "stopped", "message": f"Model server on port {port_to_stop} stopped."}


@app.post("/api/agent/start")
def api_start_agent(req: StartAgentRequest):
    global running_processes
    agent_path = os.path.join(AGENTS_DIR, req.agent_file)
    if not os.path.exists(agent_path):
        raise HTTPException(status_code=404, detail=f"Agent script '{req.agent_file}' not found in agents/ folder.")

    # Check if this exact agent is already running on the same port
    curr_agent = running_processes["agent"]
    if (
        curr_agent.get("process")
        and curr_agent["process"].poll() is None
        and curr_agent.get("agent_file") == req.agent_file
        and curr_agent.get("port") == req.port
    ):
        return {
            "status": "already_running",
            "message": f"Agent '{req.agent_file}' is already running on port {req.port}",
            "port": req.port,
        }

    # CRITICAL: Mutual exclusion - only 1 agent can run at one time!
    stop_running_agent()

    cmd = [
        PYTHON_EXEC,
        agent_path,
        "-l", str(req.port),
        "-m", str(req.model_port),
    ]
    active_model = running_processes["model"].get("model_name")
    if active_model:
        cmd.extend(["--model", active_model])

    try:
        agent_log_path = os.path.join(PROJECT_ROOT, "logs", "agent_server.log")
        os.makedirs(os.path.dirname(agent_log_path), exist_ok=True)
        with open(agent_log_path, "w", encoding="utf-8") as f:
            f.write(f"=== Starting Agent Server for {req.agent_file} on port {req.port} ===\n")

        log_fd = open(agent_log_path, "a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        log_fd.close()

        agent_data = {
            "process": proc,
            "agent_file": req.agent_file,
            "port": req.port,
            "pid": proc.pid,
            "started_at": time.time(),
            "last_exit_code": None,
            "last_error": None,
        }
        running_processes["agent"] = agent_data
        running_processes["agents"][req.agent_file] = agent_data

        return {
            "status": "started",
            "message": f"Server agent '{req.agent_file}' starting on port {req.port}",
            "port": req.port,
            "pid": proc.pid,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start agent: {e}")


@app.post("/api/agent/stop")
def api_stop_agent(req: Optional[StopAgentRequest] = None):
    target_port = req.port if (req and req.port) else running_processes["agent"].get("port", 8001)
    stopped_name = running_processes["agent"].get("agent_file") or (req.agent_file if req else "agent")
    stop_running_agent(target_port)
    return {"status": "stopped", "message": f"Agent '{stopped_name}' stopped"}


@app.post("/api/app/terminate")
def api_terminate_app(req: Optional[TerminateAppRequest] = None):
    """Terminates model serving and agent serving, freeing all GPU and process resources."""
    shutdown_web = req.shutdown_web if req else False
    stopped_services = []

    # 1. Shutdown model serving
    try:
        api_stop_model()
        stopped_services.append("Model Serving")
    except Exception as e:
        print(f"[Warning] Error stopping model server during termination: {e}", flush=True)

    # 2. Shutdown agent serving
    try:
        stop_running_agent()
        stopped_services.append("Agent Serving")
    except Exception as e:
        print(f"[Warning] Error stopping agent server during termination: {e}", flush=True)

    # 3. Terminate any lingering vLLM worker processes
    kill_lingering_vllm_processes()

    # 4. Optional web server shutdown
    if shutdown_web:
        def _delayed_exit():
            time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGINT)
        threading.Thread(target=_delayed_exit, daemon=True).start()

    return {
        "status": "terminated",
        "shutdown_web": shutdown_web,
        "message": f"Successfully shut down: {', '.join(stopped_services)}.",
        "services": stopped_services,
    }


@app.post("/api/chat")
def api_chat(req: ChatQueryRequest):
    """Routes query to active Agent Server or direct Model Server."""
    if req.target == "agent":
        # Check if agent is ready
        agent_info = running_processes["agent"]
        if agent_info.get("process") and not check_agent_server_health(req.port):
            return {
                "reply": "⏳ Agent server is still starting up. Please wait a moment until it is ready.",
                "tools_used": [],
                "status": "loading",
            }

        agent_url = f"http://127.0.0.1:{req.port}/agent/chat"
        try:
            agent_payload = {"prompt": req.prompt}
            if req.temperature is not None:
                agent_payload["temperature"] = req.temperature
            agent_headers = {"x-from-entity": "User", "x-to-entity": "Agent"}
            resp = requests.post(agent_url, json=agent_payload, headers=agent_headers, timeout=180)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "reply": data.get("reply", ""),
                    "tools_used": data.get("tools_used", []),
                    "status": "success",
                }
            else:
                return {"reply": f"Agent error ({resp.status_code}): {resp.text}", "tools_used": [], "status": "error"}
        except Exception as e:
            return {
                "reply": f"Could not connect to Agent on port {req.port}. Is the agent server running? ({e})",
                "tools_used": [],
                "status": "error",
            }
    else:
        # Direct Model Server completion
        health = check_model_server_health(req.port)
        if not health["healthy"]:
            model_info = running_processes["model"]
            if model_info.get("process") and model_info["process"].poll() is None:
                elapsed = int(time.time() - model_info.get("started_at", time.time()))
                return {
                    "reply": f"⏳ The model server is currently loading weights & warming up KV cache (~{elapsed}s). Please wait until the status indicator turns green (Actively Serving) before prompting.",
                    "tools_used": [],
                    "status": "loading",
                }
            else:
                return {
                    "reply": f"⚪ Model server on port {req.port} is not running. Please start the model server from the 'Server & Agents' tab first.",
                    "tools_used": [],
                    "status": "offline",
                }

        model_url = f"http://127.0.0.1:{req.port}/v1/chat/completions"
        headers = {
            "Authorization": "Bearer your-internal-secure-gateway-token-xyz",
            "x-from-entity": "User",
            "x-to-entity": "Model",
        }

        # Discover active models dynamically from the model server to ensure exact model match
        actual_model_id = None
        try:
            m_resp = requests.get(f"http://127.0.0.1:{req.port}/v1/models", headers=headers, timeout=2)
            if m_resp.status_code == 200:
                m_data = m_resp.json()
                if "data" in m_data and len(m_data["data"]) > 0:
                    available_ids = [m["id"] for m in m_data["data"]]
                    if req.model in available_ids:
                        actual_model_id = req.model
                    else:
                        req_base = os.path.basename(os.path.normpath(req.model)) if req.model else ""
                        if req_base in available_ids:
                            actual_model_id = req_base
                        else:
                            actual_model_id = available_ids[0]
        except Exception:
            pass

        target_model = actual_model_id or health.get("active_model") or (os.path.basename(os.path.normpath(req.model)) if req.model else None) or "Llama-3.2-3B-Instruct"
        temp = req.temperature if req.temperature is not None else 0.7

        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": req.prompt}],
            "max_tokens": 512,
            "temperature": temp,
        }
        try:
            resp = requests.post(model_url, json=payload, headers=headers, timeout=180)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return {"reply": content, "tools_used": [], "status": "success"}
            else:
                return {"reply": f"Model server error ({resp.status_code}): {resp.text}", "tools_used": [], "status": "error"}
        except Exception as e:
            return {
                "reply": f"Could not connect to Model Server on port {req.port}: {e}",
                "tools_used": [],
                "status": "error",
            }


@app.on_event("shutdown")
def on_app_shutdown():
    """Ensure all background model and agent processes are cleanly terminated when web app quits."""
    try:
        api_stop_model()
    except Exception:
        pass
    try:
        stop_running_agent()
    except Exception:
        pass
    kill_lingering_vllm_processes()


@app.get("/api/logs")
def api_logs():
    if os.path.exists(LOGS_FILE):
        try:
            with open(LOGS_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    events = json.loads(content)
                    # Exclude health check, ping, and status discovery events
                    filtered = [
                        e for e in events
                        if str(e.get("endpoint", "")).lower() not in {"/health", "/ping", "/healthz", "/v1/models"}
                        and not str(e.get("endpoint", "")).lower().startswith(("/health", "/ping"))
                    ]
                    return filtered[-100:]  # Return most recent 100 events
        except Exception:
            pass
    return []


def get_model_start_timestamp() -> Optional[float]:
    """Returns the UNIX timestamp when the model was started."""
    # 1. In-memory tracked started_at
    started_at = running_processes["model"].get("started_at")
    if started_at:
        return started_at

    # 2. File-persisted start time
    if os.path.exists(MODEL_START_TIME_FILE):
        try:
            with open(MODEL_START_TIME_FILE, "r", encoding="utf-8") as f:
                val = float(f.read().strip())
                if val > 0:
                    running_processes["model"]["started_at"] = val
                    return val
        except Exception:
            pass

    # 3. If model server is active on port 8000, retrieve process start time
    try:
        health = check_model_server_health(running_processes["model"].get("port", 8000))
        if health["healthy"]:
            try:
                out = subprocess.check_output(
                    ["lsof", "-ti", f":{running_processes['model'].get('port', 8000)}"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                pids = [int(p.strip()) for p in out.strip().split() if p.strip().isdigit()]
                if pids:
                    pid = pids[0]
                    stat_path = f"/proc/{pid}/stat"
                    if os.path.exists(stat_path):
                        with open(stat_path, "r") as sf:
                            fields = sf.read().split()
                        boot_time = 0
                        with open("/proc/stat", "r") as pf:
                            for line in pf:
                                if line.startswith("btime "):
                                    boot_time = float(line.split()[1])
                                    break
                        clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
                        proc_start = boot_time + (float(fields[21]) / clk_tck)
                        running_processes["model"]["started_at"] = proc_start
                        return proc_start
            except Exception:
                pass
    except Exception:
        pass

    return None


@app.get("/api/telemetry")
def api_telemetry(
    request: Request,
    interval: str = "15m",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    Returns:
      1. Top summary metrics (Prompts, Responses, Errors, Input Tokens, Output Tokens)
         calculated since the model started.
      2. Time-bucketed series for Requests and Tokens graphs according to interval & time range.
    """
    req_range = request.query_params.get("range") or request.query_params.get("time_range") or "1d"
    effective_range = req_range.lower()
    # 1. Parse all events
    events = []
    if os.path.exists(LOGS_FILE):
        try:
            with open(LOGS_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    raw_events = json.loads(content)
                    if isinstance(raw_events, list):
                        events = raw_events
        except Exception:
            events = []

    # 2. Filter model events (service == 'llm' and ignore non-inference endpoints)
    ignored_endpoints = {"/health", "/ping", "/healthz", "/v1/models", "/api/status"}
    model_events = []
    for e in events:
        if str(e.get("service", "")).lower() == "llm":
            ep = str(e.get("endpoint", "")).lower()
            if ep in ignored_endpoints or ep.startswith(("/health", "/ping")):
                continue
            ts_str = e.get("timestamp")
            if not ts_str:
                continue
            try:
                dt = datetime.fromisoformat(ts_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                model_events.append((dt, e))
            except Exception:
                continue

    # Sort model events by timestamp
    model_events.sort(key=lambda x: x[0])

    # 3. Calculate Top Statistics since model started
    model_start_ts = get_model_start_timestamp()
    start_dt = None
    if model_start_ts:
        start_dt = datetime.fromtimestamp(model_start_ts, tz=timezone.utc)

    # If model is running with a start_dt, count events since start_dt.
    # If no events occurred after start_dt yet or if model_start_ts is absent, compute across existing model events.
    if start_dt:
        events_after_start = [e for dt, e in model_events if dt >= start_dt]
        since_start_events = events_after_start if events_after_start else [e for _, e in model_events]
    else:
        since_start_events = [e for _, e in model_events]

    total_prompts = 0
    total_responses = 0
    total_errors = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for e in since_start_events:
        etype = str(e.get("type", "")).lower()
        if etype == "request":
            total_prompts += 1
        elif etype == "response":
            status_code = e.get("status_code", 200)
            payload = e.get("payload") or {}
            is_err = status_code >= 400 or (isinstance(payload, dict) and "error" in payload)
            if is_err:
                total_errors += 1
            else:
                total_responses += 1

            if isinstance(payload, dict):
                usage = payload.get("usage") or {}
                if isinstance(usage, dict):
                    in_tok = usage.get("prompt_tokens") or 0
                    out_tok = usage.get("completion_tokens") or 0
                    total_input_tokens += int(in_tok)
                    total_output_tokens += int(out_tok)

    # 4. Determine Time Range for Graphs
    now = datetime.now(timezone.utc)
    interval_map = {
        "1m": 60,
        "15m": 15 * 60,
        "1h": 3600,
        "1d": 86400,
    }
    interval_sec = interval_map.get(interval.lower(), 15 * 60)

    range_lower = effective_range
    if range_lower == "1h":
        t_end = now
        t_start = now - timedelta(hours=1)
    elif range_lower == "1d":
        t_end = now
        t_start = now - timedelta(days=1)
    elif range_lower in ["week", "7d"]:
        t_end = now
        t_start = now - timedelta(days=7)
    elif range_lower in ["month", "30d"]:
        t_end = now
        t_start = now - timedelta(days=30)
    elif range_lower == "custom":
        try:
            if start_date:
                if len(start_date) == 10:  # YYYY-MM-DD
                    t_start = datetime.fromisoformat(f"{start_date}T00:00:00").replace(tzinfo=timezone.utc)
                else:
                    t_start = datetime.fromisoformat(start_date)
                    if t_start.tzinfo is None:
                        t_start = t_start.replace(tzinfo=timezone.utc)
            else:
                t_start = now - timedelta(days=1)

            if end_date:
                if len(end_date) == 10:  # YYYY-MM-DD
                    t_end = datetime.fromisoformat(f"{end_date}T23:59:59").replace(tzinfo=timezone.utc)
                else:
                    t_end = datetime.fromisoformat(end_date)
                    if t_end.tzinfo is None:
                        t_end = t_end.replace(tzinfo=timezone.utc)
            else:
                t_end = now

            if t_start > t_end:
                t_start, t_end = t_end, t_start
        except Exception:
            t_start = now - timedelta(days=1)
            t_end = now
    else:
        t_start = now - timedelta(days=1)
        t_end = now

    # 5. Build Time Buckets for the Time Range
    total_duration_sec = max(interval_sec, (t_end - t_start).total_seconds())
    bucket_count = int(total_duration_sec // interval_sec) + 1
    # Cap bucket count to 500 to keep UI ultra smooth
    if bucket_count > 500:
        bucket_count = 500
        interval_sec = total_duration_sec / bucket_count

    bucket_labels = []
    bucket_prompts = [0] * bucket_count
    bucket_responses = [0] * bucket_count
    bucket_errors = [0] * bucket_count
    bucket_input_tokens = [0] * bucket_count
    bucket_output_tokens = [0] * bucket_count
    bucket_total_tokens = [0] * bucket_count

    start_ts = t_start.timestamp()
    end_ts = t_end.timestamp()

    # Determine date formatting style
    is_multi_day = (t_end - t_start).total_seconds() > 86400

    for i in range(bucket_count):
        b_time = datetime.fromtimestamp(start_ts + (i * interval_sec), tz=timezone.utc)
        if interval_sec >= 86400:
            label = b_time.strftime("%b %d")
        elif is_multi_day:
            label = b_time.strftime("%b %d %H:%M")
        else:
            label = b_time.strftime("%H:%M")
        bucket_labels.append(label)

    # Populate buckets from model events within [t_start, t_end]
    for dt, e in model_events:
        evt_ts = dt.timestamp()
        if start_ts <= evt_ts <= end_ts:
            idx = int((evt_ts - start_ts) / interval_sec)
            if 0 <= idx < bucket_count:
                etype = str(e.get("type", "")).lower()
                if etype == "request":
                    bucket_prompts[idx] += 1
                elif etype == "response":
                    status_code = e.get("status_code", 200)
                    payload = e.get("payload") or {}
                    if status_code >= 400 or (isinstance(payload, dict) and "error" in payload):
                        bucket_errors[idx] += 1
                    else:
                        bucket_responses[idx] += 1

                    if isinstance(payload, dict):
                        usage = payload.get("usage") or {}
                        if isinstance(usage, dict):
                            in_tok = int(usage.get("prompt_tokens") or 0)
                            out_tok = int(usage.get("completion_tokens") or 0)
                            tot_tok = int(usage.get("total_tokens") or (in_tok + out_tok))
                            bucket_input_tokens[idx] += in_tok
                            bucket_output_tokens[idx] += out_tok
                            bucket_total_tokens[idx] += tot_tok

    model_info = running_processes.get("model", {})
    model_name = model_info.get("model_name") or "None"
    uptime_sec = int(now.timestamp() - model_start_ts) if model_start_ts else 0

    return {
        "summary": {
            "total_prompts": total_prompts,
            "total_responses": total_responses,
            "total_errors": total_errors,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "model_name": model_name,
            "model_started_at": start_dt.isoformat() if start_dt else None,
            "uptime_seconds": uptime_sec if uptime_sec > 0 else None,
        },
        "series": {
            "labels": bucket_labels,
            "requests": bucket_prompts,
            "prompts": bucket_prompts,
            "responses": bucket_responses,
            "errors": bucket_errors,
            "input_tokens": bucket_input_tokens,
            "output_tokens": bucket_output_tokens,
            "total_tokens": bucket_total_tokens,
        },
        "query": {
            "interval": interval,
            "range": effective_range,
            "start": t_start.isoformat(),
            "end": t_end.isoformat(),
        }
    }


TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")
INDEX_HTML_PATH = os.path.join(TEMPLATES_DIR, "index.html")


def load_html_content() -> str:
    """Loads the main single-page application HTML template from disk."""
    if os.path.exists(INDEX_HTML_PATH):
        try:
            with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"<h1>Error loading index.html: {e}</h1>"
    return "<h1>Error: templates/index.html not found.</h1>"


# Maintain HTML_CONTENT attribute for backward compatibility
HTML_CONTENT = load_html_content()


# Full HTML5 Single Page Web Application with Rich Aesthetics
@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTMLResponse(content=load_html_content())


def start_web_app(port: int = 8002, host: str = "127.0.0.1"):
    """Starts the web app console server."""
    try:
        from serve_private_llm import check_and_notify_existing_instances
        check_and_notify_existing_instances(target_ports=[port, 8000, 8001])
    except Exception:
        pass
    print("\n" + "=" * 62)
    print(f"  Starting Web App Console on http://{host}:{port}")
    print("=" * 62 + "\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    port = 8002
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    start_web_app(port=port)

