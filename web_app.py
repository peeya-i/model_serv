import asyncio
import json
import os
import signal
import platform
import subprocess
import sys
import time
from typing import Any, Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
AGENTS_DIR = os.path.join(PROJECT_ROOT, "agents")
LOGS_FILE = os.path.join(PROJECT_ROOT, "logs", "events.json")
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
PYTHON_EXEC = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

app = FastAPI(title="Private LLM & Local Agent Server Console")

# In-memory tracking for background processes launched via Web App
running_processes = {
    "model": {
        "process": None,
        "model_name": None,
        "port": 8001,
        "pid": None,
        "started_at": None,
        "last_exit_code": None,
        "last_error": None,
    },
    "agent": {
        "process": None,
        "agent_file": None,
        "port": 8002,
        "pid": None,
        "started_at": None,
        "last_exit_code": None,
        "last_error": None,
    },
    "agents": {},  # agent_name -> state dict
}


def check_model_server_health(port: int = 8001) -> dict[str, Any]:
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


def check_agent_server_health(port: int = 8002) -> bool:
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


def stop_running_agent(port: int = 8002):
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


def get_available_models_list() -> list[dict[str, Any]]:
    """Lists models stored locally in models/ or discovered in cache."""
    models = []
    if os.path.isdir(MODELS_DIR):
        for entry in sorted(os.listdir(MODELS_DIR)):
            full_path = os.path.join(MODELS_DIR, entry)
            if os.path.isdir(full_path) and not entry.startswith(("models--", ".", "hub", "xet")):
                cfg_path = os.path.join(full_path, "config.json")
                if os.path.exists(cfg_path):
                    max_len = 2048 if "tinyllama" in entry.lower() else 4096
                    models.append({
                        "name": entry,
                        "path": f"models/{entry}",
                        "max_model_len": max_len,
                        "type": "Local Weights Directory",
                    })

    # Hugging Face cache discovery
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir(MODELS_DIR)
        for repo in cache_info.repos:
            repo_id = repo.repo_id
            if not any(m["name"] == repo_id or m["path"] == repo_id for m in models):
                max_len = 2048 if "tinyllama" in repo_id.lower() else 4096
                models.append({
                    "name": repo_id,
                    "path": repo_id,
                    "max_model_len": max_len,
                    "type": "Hugging Face Cache",
                })
    except Exception:
        pass

    if not models:
        models.append({
            "name": "Llama-3.2-3B-Instruct",
            "path": "models/Llama-3.2-3B-Instruct",
            "max_model_len": 4096,
            "type": "Default Local Model",
        })

    return models


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
                agent_port = active_agent.get("port", 8002) if is_running else 8002
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
    port: int = 8001


class StopModelRequest(BaseModel):
    port: Optional[int] = 8001


class StartAgentRequest(BaseModel):
    agent_file: str
    port: int = 8002
    model_port: int = 8001


class StopAgentRequest(BaseModel):
    agent_file: Optional[str] = None
    port: Optional[int] = 8002


class ChatQueryRequest(BaseModel):
    prompt: str
    target: str = "agent"  # "agent" or "model"
    port: int = 8002
    model: Optional[str] = None
    temperature: Optional[float] = 0.7


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
    model_port = model_info.get("port", 8001)
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
    agent_port = agent_info.get("port", 8002)
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

        running_processes["model"] = {
            "process": proc,
            "model_name": req.model,
            "port": req.port,
            "pid": proc.pid,
            "started_at": time.time(),
            "last_exit_code": None,
            "last_error": None,
        }
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
    port_to_stop = req.port if (req and req.port) else running_processes["model"].get("port", 8001)

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
    target_port = req.port if (req and req.port) else running_processes["agent"].get("port", 8002)
    stopped_name = running_processes["agent"].get("agent_file") or (req.agent_file if req else "agent")
    stop_running_agent(target_port)
    return {"status": "stopped", "message": f"Agent '{stopped_name}' stopped"}


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


def start_web_app(port: int = 8000, host: str = "127.0.0.1"):
    """Starts the web app console server."""
    try:
        from serve_private_llm import check_and_notify_existing_instances
        check_and_notify_existing_instances(target_ports=[port, 8001, 8002])
    except Exception:
        pass
    print("\n" + "=" * 62)
    print(f"  Starting Web App Console on http://{host}:{port}")
    print("=" * 62 + "\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    start_web_app(port=port)

