import asyncio
import json
import os
import signal
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
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                        error_msg = "".join(lines[-8:]).strip()
                except Exception:
                    pass
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
            resp = requests.post(agent_url, json={"prompt": req.prompt}, timeout=60)
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
        headers = {"Authorization": "Bearer your-internal-secure-gateway-token-xyz"}

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

        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": req.prompt}],
            "max_tokens": 512,
            "temperature": 0.2,
        }
        try:
            resp = requests.post(model_url, json=payload, headers=headers, timeout=60)
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


@app.get("/api/logs")
def api_logs():
    if os.path.exists(LOGS_FILE):
        try:
            with open(LOGS_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    events = json.loads(content)
                    return events[-100:]  # Return most recent 100 events
        except Exception:
            pass
    return []


# Full HTML5 Single Page Web Application with Rich Aesthetics
@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTML_CONTENT


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Private LLM & Local Agent Hub</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #090D16;
      --bg-surface: #111726;
      --bg-card: rgba(18, 24, 38, 0.75);
      --bg-card-hover: rgba(28, 36, 56, 0.85);
      --border: rgba(255, 255, 255, 0.08);
      --border-focus: rgba(99, 102, 241, 0.6);
      --primary: #6366F1;
      --primary-gradient: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
      --emerald: #10B981;
      --emerald-glow: rgba(16, 185, 129, 0.25);
      --amber: #F59E0B;
      --cyan: #06B6D4;
      --rose: #F43F5E;
      --text-main: #F1F5F9;
      --text-muted: #94A3B8;
      --text-subtle: #64748B;
      --glass-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.05);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: 'Inter', sans-serif;
      min-height: 100vh;
      line-height: 1.5;
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(139, 92, 246, 0.08) 0%, transparent 40%);
      background-attachment: fixed;
    }

    /* Top Navigation Tabs Header */
    header {
      background: rgba(11, 15, 25, 0.85);
      backdrop-filter: blur(20px);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 0 2rem;
    }
    .nav-container {
      max-width: 1280px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 70px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 700;
      font-size: 1.15rem;
      letter-spacing: -0.02em;
    }
    .brand-badge {
      background: var(--primary-gradient);
      width: 32px;
      height: 32px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.1rem;
      box-shadow: 0 0 15px rgba(99, 102, 241, 0.5);
    }
    .tabs-group {
      display: flex;
      background: rgba(255, 255, 255, 0.04);
      padding: 4px;
      border-radius: 12px;
      border: 1px solid var(--border);
      gap: 4px;
    }
    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 8px 18px;
      font-size: 0.9rem;
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .tab-btn:hover {
      color: var(--text-main);
      background: rgba(255, 255, 255, 0.05);
    }
    .tab-btn.active {
      background: var(--primary-gradient);
      color: #fff;
      box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
    }

    /* Main Container */
    main {
      max-width: 1280px;
      margin: 2rem auto;
      padding: 0 1.5rem 4rem;
    }

    .page-content {
      display: none;
      animation: fadeIn 0.3s ease forwards;
    }
    .page-content.active {
      display: block;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Card Panels */
    .glass-card {
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.75rem;
      margin-bottom: 2rem;
      box-shadow: var(--glass-shadow);
      transition: border-color 0.2s;
    }
    .glass-card:hover {
      border-color: rgba(255, 255, 255, 0.14);
    }

    .card-title-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1.25rem;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .card-title {
      font-size: 1.15rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .card-subtitle {
      font-size: 0.85rem;
      color: var(--text-subtle);
    }

    /* Rows on First Page */
    .config-line {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 1rem 1.25rem;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--border);
      border-radius: 12px;
      margin-bottom: 0.85rem;
      transition: all 0.2s ease;
      gap: 1rem;
    }
    .config-line:hover {
      background: var(--bg-card-hover);
      border-color: rgba(99, 102, 241, 0.3);
    }
    .line-info {
      flex: 1;
    }
    .line-title {
      font-weight: 600;
      font-size: 0.95rem;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .line-desc {
      font-size: 0.8rem;
      color: var(--text-subtle);
      margin-top: 2px;
    }

    .line-actions {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .port-box {
      display: flex;
      align-items: center;
      gap: 6px;
      background: rgba(0, 0, 0, 0.3);
      padding: 6px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    .port-label {
      font-size: 0.75rem;
      color: var(--text-subtle);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .port-input {
      background: transparent;
      border: none;
      color: var(--cyan);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.9rem;
      font-weight: 600;
      width: 60px;
      outline: none;
    }

    /* Buttons */
    .btn {
      padding: 8px 18px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.85rem;
      border: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }
    .btn-enable {
      background: rgba(16, 185, 129, 0.15);
      color: #34D399;
      border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .btn-enable:hover {
      background: rgba(16, 185, 129, 0.3);
      box-shadow: 0 0 15px var(--emerald-glow);
    }
    .btn-disable {
      background: rgba(244, 63, 94, 0.15);
      color: #FB7185;
      border: 1px solid rgba(244, 63, 94, 0.4);
    }
    .btn-disable:hover {
      background: rgba(244, 63, 94, 0.3);
    }
    .btn-loading {
      background: rgba(245, 158, 11, 0.2);
      border: 1px solid rgba(245, 158, 11, 0.4);
      color: #FCD34D;
      cursor: pointer;
    }
    .btn-loading:hover {
      background: rgba(245, 158, 11, 0.3);
      box-shadow: 0 4px 15px rgba(245, 158, 11, 0.25);
    }
    .btn-primary {
      background: var(--primary-gradient);
      color: #fff;
    }
    .btn-primary:hover {
      filter: brightness(1.15);
      box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }

    /* Pulse Status Dots */
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 20px;
      transition: all 0.3s ease;
    }
    .status-badge.active {
      background: rgba(16, 185, 129, 0.15);
      color: #34D399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-badge.loading {
      background: rgba(245, 158, 11, 0.15);
      color: #FBBF24;
      border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .status-badge.inactive {
      background: rgba(100, 116, 139, 0.15);
      color: var(--text-subtle);
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .status-badge.error {
      background: rgba(244, 63, 94, 0.15);
      color: #FB7185;
      border: 1px solid rgba(244, 63, 94, 0.35);
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }
    .dot.active {
      background: #10B981;
      box-shadow: 0 0 8px #10B981;
      animation: pulse 2s infinite;
    }
    .dot.loading {
      background: #F59E0B;
      box-shadow: 0 0 10px #F59E0B;
      animation: pulse-fast 1s infinite;
    }
    .dot.inactive {
      background: #64748B;
    }
    .dot.error {
      background: #F43F5E;
      box-shadow: 0 0 8px #F43F5E;
    }
    @keyframes pulse {
      0%, 100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.3); opacity: 0.7; }
    }
    @keyframes pulse-fast {
      0%, 100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.4); opacity: 0.5; }
    }

    /* Custom model input */
    .model-input-group {
      display: flex;
      gap: 10px;
      margin-top: 10px;
      align-items: center;
    }
    .text-input {
      flex: 1;
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 9px 14px;
      color: var(--text-main);
      font-size: 0.9rem;
      outline: none;
      transition: border-color 0.2s;
    }
    .text-input:focus {
      border-color: var(--primary);
    }
    select.text-input option {
      background: #111726;
      color: #F1F5F9;
    }

    /* GPU Hardware Cards Grid */
    .gpu-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-top: 1rem;
    }
    .gpu-stat-box {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.1rem;
      position: relative;
      overflow: hidden;
    }
    .gpu-stat-box::before {
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
      background: var(--primary-gradient);
    }
    .gpu-label {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      color: var(--text-subtle);
      letter-spacing: 0.05em;
    }
    .gpu-value {
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--text-main);
      margin-top: 4px;
      word-break: break-word;
    }
    .gpu-subvalue {
      font-size: 0.75rem;
      color: var(--cyan);
      margin-top: 4px;
    }

    /* VRAM Progress Bar */
    .progress-bar-wrap {
      width: 100%;
      height: 8px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 4px;
      overflow: hidden;
      margin-top: 8px;
    }
    .progress-bar {
      height: 100%;
      background: linear-gradient(90deg, #10B981, #F59E0B);
      border-radius: 4px;
      transition: width 0.5s ease;
    }

    /* Chat Tab */
    .chat-container {
      display: flex;
      flex-direction: column;
      height: 600px;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
    }
    .chat-header {
      padding: 1rem 1.5rem;
      background: rgba(0, 0, 0, 0.3);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .chat-messages {
      flex: 1;
      padding: 1.5rem;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    .message {
      max-width: 80%;
      padding: 0.9rem 1.25rem;
      border-radius: 14px;
      font-size: 0.92rem;
      line-height: 1.6;
    }
    .message.user {
      align-self: flex-end;
      background: var(--primary-gradient);
      color: #fff;
      border-bottom-right-radius: 4px;
    }
    .message.assistant {
      align-self: flex-start;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border);
      color: var(--text-main);
      border-bottom-left-radius: 4px;
    }
    .chat-input-area {
      padding: 1rem 1.5rem;
      border-top: 1px solid var(--border);
      display: flex;
      gap: 10px;
      background: rgba(0, 0, 0, 0.25);
    }

    /* Logs Table */
    .log-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }
    .log-table th {
      text-align: left;
      padding: 10px 14px;
      background: rgba(0, 0, 0, 0.3);
      color: var(--text-subtle);
      font-weight: 600;
      border-bottom: 1px solid var(--border);
    }
    .log-table td {
      padding: 10px 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      font-family: 'JetBrains Mono', monospace;
    }
    .log-table tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }
    .badge-srv {
      padding: 2px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
    }
    .badge-llm { background: rgba(99, 102, 241, 0.2); color: #818CF8; }
    .badge-agent { background: rgba(16, 185, 129, 0.2); color: #34D399; }
    .badge-langchain { background: rgba(245, 158, 11, 0.2); color: #FBBF24; }

    .json-preview {
      background: #060910;
      padding: 8px 12px;
      border-radius: 6px;
      max-height: 80px;
      overflow-y: auto;
      font-size: 0.75rem;
      color: #A5B4FC;
    }
  </style>
</head>
<body>

  <!-- Top Navigation with 3 Tabs -->
  <header>
    <div class="nav-container">
      <div class="brand">
        <div class="brand-badge">⚡</div>
        <span>Private LLM & Agent Hub</span>
      </div>

      <nav class="tabs-group">
        <button id="tab-btn-dashboard" class="tab-btn active" onclick="switchTab('dashboard')">
          <span>⚙️</span> Server & Agents
        </button>
        <button id="tab-btn-chat" class="tab-btn" onclick="switchTab('chat')">
          <span>💬</span> Chat Console
        </button>
        <button id="tab-btn-logs" class="tab-btn" onclick="switchTab('logs')">
          <span>📋</span> Event Logs
        </button>
      </nav>

      <div style="display:flex; align-items:center; gap:8px;">
        <div id="top-model-status-pill" class="status-badge inactive">
          <span id="top-model-dot" class="dot inactive"></span>
          <span id="top-model-text">LLM: Stopped</span>
        </div>
        <div id="system-status-pill" class="status-badge active">
          <span class="dot active"></span>
          <span>Web Console Active</span>
        </div>
      </div>
    </div>
  </header>

  <main>
    <!-- PAGE 1: Server Side Agents & Model Selection & GPU -->
    <div id="page-dashboard" class="page-content active">
      
      <!-- 1. Server-side Agent Dropdown Selection (Only 1 can be running at one time) -->
      <section class="glass-card">
        <div class="card-title-row">
          <div>
            <h2 class="card-title">🤖 Server-Side Agent Selection</h2>
            <p class="card-subtitle">Choose a server agent from <code style="color:var(--cyan)">agents/</code> (Only 1 agent can run at one time)</p>
          </div>
          <div id="agent-status-badge" class="status-badge inactive">
            <span id="agent-dot" class="dot inactive"></span>
            <span id="agent-status-text">No Agent Running</span>
          </div>
        </div>

        <div class="config-line" style="flex-wrap: wrap; gap: 1rem; align-items: flex-end;">
          <div style="flex: 1; min-width: 280px;">
            <label for="agent-select" style="display:block; font-size: 0.75rem; color: var(--text-subtle); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
              Agent Selection (Dropdown)
            </label>
            <select id="agent-select" class="text-input" style="width: 100%; font-family: 'JetBrains Mono', monospace;" onchange="onAgentSelectChange()">
              <option value="">Scanning agents/ folder...</option>
            </select>
            <div id="agent-desc-text" style="font-size: 0.8rem; color: var(--text-subtle); margin-top: 6px;">
              Loading agent microservice details...
            </div>
          </div>

          <div class="line-actions" style="display: flex; align-items: center; gap: 1rem;">
            <div class="port-box">
              <span class="port-label">Port</span>
              <input type="number" id="agent-port-input" class="port-input" value="8002">
            </div>
            <button id="btn-agent-toggle" class="btn btn-enable" onclick="toggleSelectedAgent()">
              <span>▶</span> Enable Agent
            </button>
          </div>
        </div>
      </section>

      <!-- 2. Models Selection Dropdown & Text Box -->
      <section class="glass-card">
        <div class="card-title-row">
          <div>
            <h2 class="card-title">🧠 Local Model Inference Engine</h2>
            <p class="card-subtitle">Choose model from dropdown with option to enter or edit model name in text box</p>
          </div>
          <div id="model-status-indicator" class="status-badge inactive">
            <span id="model-dot" class="dot inactive"></span>
            <span id="model-status-text">Server Stopped</span>
          </div>
        </div>

        <div class="config-line" style="flex-direction: column; align-items: stretch; gap: 1rem;">
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem;">
            <!-- Dropdown choice -->
            <div>
              <label for="model-select" style="display:block; font-size: 0.75rem; color: var(--text-subtle); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
                Model Selection (Dropdown)
              </label>
              <select id="model-select" class="text-input" style="width: 100%; font-family: 'JetBrains Mono', monospace;" onchange="onModelSelectChange()">
                <option value="">Scanning models/ folder...</option>
              </select>
            </div>

            <!-- Text box to view or enter model name -->
            <div>
              <label for="model-text-input" style="display:block; font-size: 0.75rem; color: var(--text-subtle); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
                Model Name / Path (Text Box)
              </label>
              <input type="text" id="model-text-input" class="text-input" style="width: 100%; font-family: 'JetBrains Mono', monospace;" placeholder="e.g. models/Llama-3.2-3B-Instruct or Hugging Face repo" oninput="onModelTextInput()">
            </div>
          </div>

          <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem; padding-top: 0.5rem; border-top: 1px solid rgba(255, 255, 255, 0.05);">
            <div id="model-spec-info" style="font-size: 0.8rem; color: var(--cyan);">
              Ready to serve local LLM inference
            </div>

            <div class="line-actions" style="display: flex; align-items: center; gap: 1rem;">
              <div class="port-box">
                <span class="port-label">Port</span>
                <input type="number" id="model-port-input" class="port-input" value="8001">
              </div>
              <button id="btn-model-toggle" class="btn btn-enable" onclick="toggleModelServer()">
                <span>▶</span> Enable Model
              </button>
            </div>
          </div>
        </div>
      </section>

      <!-- 3. GPU Model Number and Capabilities (Below Model Selection) -->
      <section class="glass-card" style="border-color: rgba(6, 182, 212, 0.2);">
        <div class="card-title-row">
          <div>
            <h2 class="card-title">⚡ GPU Hardware & Compute Capabilities</h2>
            <p class="card-subtitle">Real-time hardware inspection and available VRAM metrics</p>
          </div>
          <span style="font-size:0.8rem; color:var(--emerald); font-weight:600;">Hardware Accelerated</span>
        </div>

        <div class="gpu-grid">
          <!-- GPU Model Number -->
          <div class="gpu-stat-box">
            <div class="gpu-label">GPU Model Name / Number</div>
            <div id="gpu-model-val" class="gpu-value">Loading...</div>
            <div id="gpu-driver-val" class="gpu-subvalue">Driver: ...</div>
          </div>

          <!-- Compute Capabilities -->
          <div class="gpu-stat-box">
            <div class="gpu-label">Architecture & Capabilities</div>
            <div id="gpu-cap-val" class="gpu-value" style="font-size: 0.95rem;">Loading...</div>
            <div class="gpu-subvalue">Triton & PagedAttention Active</div>
          </div>

          <!-- Number of Processors -->
          <div class="gpu-stat-box">
            <div class="gpu-label">Number of Processors</div>
            <div id="gpu-proc-val" class="gpu-value" style="font-size: 0.95rem;">Loading...</div>
            <div class="gpu-subvalue">Parallel CUDA execution cores</div>
          </div>

          <!-- Number of GPUs -->
          <div class="gpu-stat-box">
            <div class="gpu-label">Number of GPUs</div>
            <div id="gpu-count-val" class="gpu-value">1 GPU</div>
            <div class="gpu-subvalue">Direct PCIe Device</div>
          </div>

          <!-- RAM Available on GPU -->
          <div class="gpu-stat-box" style="grid-column: span 2;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <div class="gpu-label">RAM Available on GPU</div>
              <span id="vram-percent-text" style="font-size:0.8rem; font-weight:700; color:var(--emerald);">--% Free</span>
            </div>
            <div id="gpu-ram-val" class="gpu-value">-- MiB / -- MiB</div>
            <div class="progress-bar-wrap">
              <div id="vram-meter" class="progress-bar" style="width: 85%;"></div>
            </div>
            <div id="gpu-ram-sub" class="gpu-subvalue">3GB CPU offload enabled for models exceeding VRAM</div>
          </div>
        </div>
      </section>

    </div>

    <!-- PAGE 2: Interactive Chat Console -->
    <div id="page-chat" class="page-content">
      <div class="chat-container">
        <div class="chat-header">
          <div style="display:flex; align-items:center; gap:12px;">
            <span style="font-weight:700;">💬 Interactive AI Playground</span>
            <div id="chat-target-status-badge" class="status-badge inactive">
              <span id="chat-target-dot" class="dot inactive"></span>
              <span id="chat-target-status-text">Checking status...</span>
            </div>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-size:0.8rem; color:var(--text-subtle);">Target:</span>
            <select id="chat-target-select" class="text-input" style="padding:4px 10px; font-size:0.8rem; width:170px;" onchange="updateChatStatusUI()">
              <option value="model">Direct Model (:8001)</option>
              <option value="agent">Server Agent (:8002)</option>
            </select>
          </div>
        </div>

        <div id="chat-messages" class="chat-messages">
          <div class="message assistant">
            Hello! I am your local AI inference assistant. You can prompt me directly or have the server agent execute local Python tools (calculator, system diagnostics, and timestamp tools).
          </div>
        </div>

        <div class="chat-input-area">
          <input type="text" id="chat-user-input" class="text-input" placeholder="Type a query, e.g. 'Calculate 25 * 40' or 'Explain private LLM deployment'..." onkeydown="if(event.key==='Enter') sendChat()">
          <button class="btn btn-primary" onclick="sendChat()">Send Prompt</button>
        </div>
      </div>
    </div>

    <!-- PAGE 3: Live Event Logs -->
    <div id="page-logs" class="page-content">
      <section class="glass-card">
        <div class="card-title-row">
          <div>
            <h2 class="card-title">📋 Structured Activity & Event Logs</h2>
            <p class="card-subtitle">Real-time requests and responses stored in <code style="color:var(--cyan)">logs/events.json</code> (max 200MB FIFO capped)</p>
          </div>
          <div style="display:flex; gap:8px;">
            <button class="btn btn-primary" style="padding:6px 12px; font-size:0.8rem;" onclick="loadLogs()">↻ Refresh Logs</button>
          </div>
        </div>

        <div style="overflow-x: auto;">
          <table class="log-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Service</th>
                <th>Type</th>
                <th>Endpoint / Route</th>
                <th>Status / Duration</th>
                <th>Complete Payload Preview</th>
              </tr>
            </thead>
            <tbody id="logs-tbody">
              <tr><td colspan="6" style="text-align:center; color:var(--text-subtle); padding:2rem;">Loading events from events.json...</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>

  </main>

  <script>
    // Tab Switching Logic
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.page-content').forEach(p => p.classList.remove('active'));

      const btn = document.getElementById(`tab-btn-${tabId}`);
      const page = document.getElementById(`page-${tabId}`);
      if (btn && page) {
        btn.classList.add('active');
        page.classList.add('active');
      }

      if (tabId === 'logs') {
        loadLogs();
      }
    }

    let currentStatusData = null;
    let availableAgents = [];
    let availableModels = [];

    // Fetch Status and GPU Info
    let refreshTimer = null;

    async function refreshStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        currentStatusData = data;
        
        // 1. Update GPU Panel
        if (data.gpu) {
          document.getElementById('gpu-model-val').textContent = data.gpu.model || "NVIDIA GPU";
          document.getElementById('gpu-driver-val').textContent = `Driver: ${data.gpu.driver_version} | CUDA: ${data.gpu.cuda_version}`;
          document.getElementById('gpu-cap-val').textContent = data.gpu.capabilities || "N/A";
          document.getElementById('gpu-proc-val').textContent = data.gpu.processors || "N/A";
          document.getElementById('gpu-count-val').textContent = `${data.gpu.num_gpus} GPU Device(s)`;

          const freeMB = data.gpu.ram_available_mb;
          const totalMB = data.gpu.ram_total_mb;
          document.getElementById('gpu-ram-val').textContent = `${freeMB} MiB Free / ${totalMB} MiB Total`;
          if (totalMB > 0) {
            const pct = Math.round((freeMB / totalMB) * 100);
            document.getElementById('vram-percent-text').textContent = `${pct}% Available`;
            document.getElementById('vram-meter').style.width = `${pct}%`;
          }
        }

        // 2. Update Server Agent Dropdown & Status
        availableAgents = data.agents || [];
        const agentSelect = document.getElementById('agent-select');
        const activeAgent = data.active_agent || { status: 'stopped', running: false };

        const prevAgentVal = agentSelect.value;
        const agentOptionsHtml = availableAgents.map(ag => {
          const isRun = ag.is_running;
          const statusTag = isRun ? (ag.status === 'ready' ? ' (● ACTIVE)' : ' (⏳ STARTING)') : '';
          return `<option value="${ag.filename}">${ag.filename}${statusTag}</option>`;
        }).join('');

        if (agentSelect.innerHTML !== agentOptionsHtml) {
          agentSelect.innerHTML = agentOptionsHtml;
          if (activeAgent.is_alive && activeAgent.agent_file) {
            agentSelect.value = activeAgent.agent_file;
          } else if (prevAgentVal && availableAgents.some(a => a.filename === prevAgentVal)) {
            agentSelect.value = prevAgentVal;
          }
        }

        updateAgentUI();

        // 3. Update Models Dropdown & Text Box & Status
        availableModels = data.models || [];
        const modelSelect = document.getElementById('model-select');
        const modelTextInput = document.getElementById('model-text-input');
        const modelServer = data.model_server || { status: 'stopped', running: false };

        let modelOptionsHtml = availableModels.map(m => {
          return `<option value="${m.path}">${m.name} (${m.type})</option>`;
        }).join('');
        modelOptionsHtml += `<option value="__custom__">✏️ Custom Model (Enter in text box)...</option>`;

        if (modelSelect.dataset.initialized !== "true") {
          modelSelect.innerHTML = modelOptionsHtml;
          modelSelect.dataset.initialized = "true";
          if (modelServer.is_alive && modelServer.model_name) {
            modelTextInput.value = modelServer.model_name;
            syncDropdownWithText(modelServer.model_name);
          } else if (availableModels.length > 0) {
            modelSelect.selectedIndex = 0;
            modelTextInput.value = availableModels[0].path;
          }
          onModelSelectChange(false);
        }

        updateModelUI();
        updateChatStatusUI();

        // Dynamically adjust polling frequency: poll rapidly (1.5s) when loading, relax (6s) when stable
        const modelLoading = data.model_server?.status === 'loading';
        const agentLoading = data.active_agent?.status === 'loading';
        const nextDelay = (modelLoading || agentLoading) ? 1500 : 6000;

        if (refreshTimer) clearTimeout(refreshTimer);
        refreshTimer = setTimeout(refreshStatus, nextDelay);

      } catch (e) {
        console.error("Status update error:", e);
        if (refreshTimer) clearTimeout(refreshTimer);
        refreshTimer = setTimeout(refreshStatus, 6000);
      }
    }

    function updateAgentUI() {
      if (!currentStatusData) return;
      const activeAgent = currentStatusData.active_agent || { status: 'stopped', running: false };
      const agentSelect = document.getElementById('agent-select');
      const selectedAgentFile = agentSelect.value;
      const descEl = document.getElementById('agent-desc-text');
      const badgeEl = document.getElementById('agent-status-badge');
      const dotEl = document.getElementById('agent-dot');
      const textEl = document.getElementById('agent-status-text');
      const btnEl = document.getElementById('btn-agent-toggle');
      const portInput = document.getElementById('agent-port-input');

      const foundAgent = availableAgents.find(a => a.filename === selectedAgentFile);
      if (foundAgent) {
        descEl.textContent = foundAgent.description;
      }

      const status = activeAgent.status || (activeAgent.running ? 'ready' : 'stopped');
      const elapsed = activeAgent.elapsed_sec !== undefined && activeAgent.elapsed_sec !== null ? `${activeAgent.elapsed_sec}s` : '';

      if (status === 'ready') {
        badgeEl.className = 'status-badge active';
        dotEl.className = 'dot active';
        textEl.textContent = `Active: ${activeAgent.agent_file} (:${activeAgent.port}, PID: ${activeAgent.pid || 'Active'})`;

        if (selectedAgentFile === activeAgent.agent_file) {
          btnEl.className = 'btn btn-disable';
          btnEl.innerHTML = '<span>⏹</span> Disable Agent';
          portInput.value = activeAgent.port;
        } else {
          btnEl.className = 'btn btn-primary';
          btnEl.innerHTML = '<span>🔄</span> Switch to this Agent';
        }
      } else if (status === 'loading') {
        badgeEl.className = 'status-badge loading';
        dotEl.className = 'dot loading';
        textEl.textContent = `Starting Microservice (${elapsed})...`;
        btnEl.className = 'btn btn-loading';
        btnEl.innerHTML = `<span>⏳</span> Starting (${elapsed})... Click to Abort`;
      } else if (status === 'crashed') {
        badgeEl.className = 'status-badge error';
        dotEl.className = 'dot error';
        textEl.textContent = 'Agent Exited Unexpectedly';
        btnEl.className = 'btn btn-enable';
        btnEl.innerHTML = '<span>▶</span> Restart Agent';
      } else {
        badgeEl.className = 'status-badge inactive';
        dotEl.className = 'dot inactive';
        textEl.textContent = 'No Agent Running';
        btnEl.className = 'btn btn-enable';
        btnEl.innerHTML = '<span>▶</span> Enable Agent';
      }
    }

    function onAgentSelectChange() {
      updateAgentUI();
    }

    async function toggleSelectedAgent() {
      const agentSelect = document.getElementById('agent-select');
      const selectedAgentFile = agentSelect.value;
      if (!selectedAgentFile) {
        alert("Please select an agent from the dropdown.");
        return;
      }

      const activeAgent = currentStatusData?.active_agent || { status: 'stopped', running: false };
      const port = parseInt(document.getElementById('agent-port-input').value) || 8002;
      const modelPort = parseInt(document.getElementById('model-port-input').value) || 8001;

      if (activeAgent.is_alive && activeAgent.agent_file === selectedAgentFile) {
        // Disable running or loading agent
        const res = await fetch('/api/agent/stop', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ agent_file: selectedAgentFile })
        });
        const d = await res.json();
        alert(d.message || "Agent stopped.");
      } else {
        // Enable selected agent (terminates any previously running agent so only 1 runs)
        const res = await fetch('/api/agent/start', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ agent_file: selectedAgentFile, port: port, model_port: modelPort })
        });
        const d = await res.json();
        alert(d.message || "Agent started.");
      }
      refreshStatus();
    }

    function updateModelUI() {
      if (!currentStatusData) return;
      const modelServer = currentStatusData.model_server || { status: 'stopped', running: false };
      const modelBadge = document.getElementById('model-status-indicator');
      const modelDot = document.getElementById('model-dot');
      const modelStatusText = document.getElementById('model-status-text');
      const modelToggleBtn = document.getElementById('btn-model-toggle');
      const specInfo = document.getElementById('model-spec-info');
      const topModelPill = document.getElementById('top-model-status-pill');
      const topModelDot = document.getElementById('top-model-dot');
      const topModelText = document.getElementById('top-model-text');

      const status = modelServer.status || (modelServer.running ? 'ready' : 'stopped');
      const elapsed = (modelServer.elapsed_sec !== undefined && modelServer.elapsed_sec !== null) ? `${modelServer.elapsed_sec}s` : '';
      const modelName = modelServer.model_name || 'Model';

      if (status === 'ready') {
        modelBadge.className = 'status-badge active';
        modelDot.className = 'dot active';
        modelStatusText.textContent = `Actively Serving on :${modelServer.port} (PID: ${modelServer.pid || 'Active'})`;
        modelToggleBtn.className = 'btn btn-disable';
        modelToggleBtn.innerHTML = '<span>⏹</span> Stop Model Server';
        if (specInfo) {
          specInfo.innerHTML = `<span style="color:var(--emerald); font-weight:600;">● Actively Serving:</span> <strong>${escapeHtml(modelName)}</strong> | REST API: <code style="color:var(--cyan)">http://127.0.0.1:${modelServer.port}/v1</code>`;
        }
        if (topModelPill) {
          topModelPill.className = 'status-badge active';
          topModelDot.className = 'dot active';
          topModelText.textContent = 'LLM: Actively Serving';
        }
      } else if (status === 'loading') {
        modelBadge.className = 'status-badge loading';
        modelDot.className = 'dot loading';
        modelStatusText.textContent = `Initializing & Loading Weights${elapsed ? ' (~' + elapsed + ')' : '...'}`;
        modelToggleBtn.className = 'btn btn-loading';
        modelToggleBtn.innerHTML = `<span>⏳</span> Starting${elapsed ? ' (' + elapsed + ')' : ''}... (Click to Abort)`;
        if (specInfo) {
          specInfo.innerHTML = `<span style="color:#FBBF24; font-weight:600;">⏳ Loading weights & initializing KV cache${elapsed ? ' (~' + elapsed + ')' : ''}...</span> Server is compiling CUDA kernels and warming up. Will turn green the moment it is actively ready.`;
        }
        if (topModelPill) {
          topModelPill.className = 'status-badge loading';
          topModelDot.className = 'dot loading';
          topModelText.textContent = `LLM: Loading${elapsed ? ' (' + elapsed + ')' : '...'}`;
        }
      } else if (status === 'crashed') {
        modelBadge.className = 'status-badge error';
        modelDot.className = 'dot error';
        modelStatusText.textContent = `Process Terminated (Exit: ${modelServer.exit_code})`;
        modelToggleBtn.className = 'btn btn-enable';
        modelToggleBtn.innerHTML = '<span>🔄</span> Enable Model';
        if (specInfo) {
          const errPreview = modelServer.last_error ? escapeHtml(modelServer.last_error.slice(-140)) : 'Process exited unexpectedly. Check logs.';
          specInfo.innerHTML = `<span style="color:var(--rose); font-weight:600;">❌ Server stopped with error:</span> ${errPreview}`;
        }
        if (topModelPill) {
          topModelPill.className = 'status-badge error';
          topModelDot.className = 'dot error';
          topModelText.textContent = 'LLM: Error';
        }
      } else {
        modelBadge.className = 'status-badge inactive';
        modelDot.className = 'dot inactive';
        modelStatusText.textContent = 'Server Stopped';
        modelToggleBtn.className = 'btn btn-enable';
        modelToggleBtn.innerHTML = '<span>▶</span> Enable Model';
        if (topModelPill) {
          topModelPill.className = 'status-badge inactive';
          topModelDot.className = 'dot inactive';
          topModelText.textContent = 'LLM: Stopped';
        }
      }
    }

    function updateChatStatusUI() {
      if (!currentStatusData) return;
      const target = document.getElementById('chat-target-select').value;
      const chatBadge = document.getElementById('chat-target-status-badge');
      const chatDot = document.getElementById('chat-target-dot');
      const chatText = document.getElementById('chat-target-status-text');
      if (!chatBadge || !chatDot || !chatText) return;

      if (target === 'model') {
        const modelServer = currentStatusData.model_server || { status: 'stopped', running: false };
        const status = modelServer.status || (modelServer.running ? 'ready' : 'stopped');
        const elapsed = (modelServer.elapsed_sec !== undefined && modelServer.elapsed_sec !== null) ? `${modelServer.elapsed_sec}s` : '';
        if (status === 'ready') {
          chatBadge.className = 'status-badge active';
          chatDot.className = 'dot active';
          chatText.textContent = `Direct Model Ready (:${modelServer.port})`;
        } else if (status === 'loading') {
          chatBadge.className = 'status-badge loading';
          chatDot.className = 'dot loading';
          chatText.textContent = `Model Loading Weights${elapsed ? ' (~' + elapsed + ')' : '...'}`;
        } else if (status === 'crashed') {
          chatBadge.className = 'status-badge error';
          chatDot.className = 'dot error';
          chatText.textContent = 'Model Server Crashed';
        } else {
          chatBadge.className = 'status-badge inactive';
          chatDot.className = 'dot inactive';
          chatText.textContent = 'Model Server Offline';
        }
      } else {
        const activeAgent = currentStatusData.active_agent || { status: 'stopped', running: false };
        const status = activeAgent.status || (activeAgent.running ? 'ready' : 'stopped');
        if (status === 'ready') {
          chatBadge.className = 'status-badge active';
          chatDot.className = 'dot active';
          chatText.textContent = `Agent Ready (:${activeAgent.port})`;
        } else if (status === 'loading') {
          chatBadge.className = 'status-badge loading';
          chatDot.className = 'dot loading';
          chatText.textContent = 'Agent Starting Up...';
        } else {
          chatBadge.className = 'status-badge inactive';
          chatDot.className = 'dot inactive';
          chatText.textContent = 'Agent Server Offline';
        }
      }
    }

    function onModelSelectChange(syncText = true) {
      const modelSelect = document.getElementById('model-select');
      const modelTextInput = document.getElementById('model-text-input');
      const specInfo = document.getElementById('model-spec-info');

      if (modelSelect.value === '__custom__') {
        specInfo.textContent = "Custom user-defined model or remote Hugging Face repository";
        if (syncText) {
          modelTextInput.focus();
        }
      } else {
        if (syncText) {
          modelTextInput.value = modelSelect.value;
        }
        const found = availableModels.find(m => m.path === modelSelect.value);
        if (found) {
          specInfo.textContent = `Type: ${found.type} | Max Context: ${found.max_model_len} tokens | Path: ${found.path}`;
        } else {
          specInfo.textContent = `Model Target: ${modelSelect.value}`;
        }
      }
    }

    function onModelTextInput() {
      const text = document.getElementById('model-text-input').value.trim();
      syncDropdownWithText(text);
    }

    function syncDropdownWithText(text) {
      const modelSelect = document.getElementById('model-select');
      const specInfo = document.getElementById('model-spec-info');
      const match = availableModels.find(m => m.path === text || m.name === text);
      if (match) {
        modelSelect.value = match.path;
        specInfo.textContent = `Type: ${match.type} | Max Context: ${match.max_model_len} tokens | Path: ${match.path}`;
      } else {
        modelSelect.value = '__custom__';
        specInfo.textContent = `Custom Model Target: ${text || "Please enter model name"}`;
      }
    }

    async function toggleModelServer() {
      const modelServer = currentStatusData?.model_server || { status: 'stopped', running: false };
      const status = modelServer.status || (modelServer.running ? 'ready' : 'stopped');
      const portInput = parseInt(document.getElementById('model-port-input').value) || 8001;
      const modelText = document.getElementById('model-text-input').value.trim();
      const modelToUse = modelText || document.getElementById('model-select').value || "models/Llama-3.2-3B-Instruct";

      if (status === 'ready' || status === 'loading') {
        const res = await fetch('/api/model/stop', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ port: portInput })
        });
        const d = await res.json();
        // Immediately update state locally for instant UI flip
        if (currentStatusData && currentStatusData.model_server) {
          currentStatusData.model_server.status = 'stopped';
          currentStatusData.model_server.running = false;
          currentStatusData.model_server.is_alive = false;
        }
        updateModelUI();
        updateChatStatusUI();
      } else {
        const res = await fetch('/api/model/start', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ model: modelToUse, port: portInput })
        });
        const d = await res.json();
        if (d.status === "already_running") {
          alert(d.message);
        } else {
          if (currentStatusData && currentStatusData.model_server) {
            currentStatusData.model_server.status = 'loading';
            currentStatusData.model_server.running = false;
            currentStatusData.model_server.is_alive = true;
            currentStatusData.model_server.elapsed_sec = 0;
          }
          updateModelUI();
          updateChatStatusUI();
        }
      }
      setTimeout(refreshStatus, 350);
    }

    async function sendChat() {
      const input = document.getElementById('chat-user-input');
      const text = input.value.trim();
      if (!text) return;

      const target = document.getElementById('chat-target-select').value;
      const agentPort = parseInt(document.getElementById('agent-port-input').value) || 8002;
      const modelPort = parseInt(document.getElementById('model-port-input').value) || 8001;
      const targetPort = target === 'agent' ? agentPort : modelPort;

      // Check if target is ready before sending
      if (target === 'model') {
        const modelServer = currentStatusData?.model_server || { status: 'stopped' };
        if (modelServer.status === 'loading') {
          const elapsed = modelServer.elapsed_sec ? ` (~${modelServer.elapsed_sec}s)` : '';
          alert(`The model server is still loading weights and warming up${elapsed}. Please wait until the status indicator turns green (Actively Serving).`);
          return;
        } else if (modelServer.status === 'stopped' || modelServer.status === 'crashed') {
          alert("The model server is currently offline. Please start it on the 'Server & Agents' tab first.");
          return;
        }
      } else {
        const activeAgent = currentStatusData?.active_agent || { status: 'stopped' };
        if (activeAgent.status === 'loading') {
          alert("The agent microservice is still initializing. Please wait a moment.");
          return;
        } else if (activeAgent.status === 'stopped' || activeAgent.status === 'crashed') {
          alert("The agent microservice is not running. Please start it on the 'Server & Agents' tab first.");
          return;
        }
      }

      const chatBox = document.getElementById('chat-messages');
      chatBox.innerHTML += `<div class="message user">${escapeHtml(text)}</div>`;
      input.value = '';
      chatBox.scrollTop = chatBox.scrollHeight;

      const loadingMsg = document.createElement('div');
      loadingMsg.className = 'message assistant';
      loadingMsg.textContent = 'Reasoning and querying...';
      chatBox.appendChild(loadingMsg);
      chatBox.scrollTop = chatBox.scrollHeight;

      try {
        const modelText = document.getElementById('model-text-input')?.value?.trim();
        const modelSelect = document.getElementById('model-select')?.value;
        const modelToUse = modelText || (modelSelect !== '__custom__' ? modelSelect : '');

        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ prompt: text, target: target, port: targetPort, model: modelToUse })
        });
        const d = await res.json();
        let toolsInfo = '';
        if (d.tools_used && d.tools_used.length > 0) {
          toolsInfo = `<div style="margin-top:6px; font-size:0.75rem; color:var(--emerald);">🛠 Tools executed: ${d.tools_used.join(', ')}</div>`;
        }
        loadingMsg.innerHTML = `${escapeHtml(d.reply || "No response received.")}${toolsInfo}`;
      } catch (e) {
        loadingMsg.innerHTML = `<span style="color:var(--rose);">Error connecting to ${target} on port ${targetPort}: ${e.message}</span>`;
      }
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function loadLogs() {
      try {
        const res = await fetch('/api/logs');
        const events = await res.json();
        const tbody = document.getElementById('logs-tbody');
        tbody.innerHTML = '';
        if (!events || events.length === 0) {
          tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color:var(--text-subtle); padding:2rem;">No events recorded yet in events.json.</td></tr>';
          return;
        }

        events.slice().reverse().forEach(e => {
          let srvBadge = 'badge-llm';
          if (e.service === 'agent_server') srvBadge = 'badge-agent';
          if (e.service === 'langchain_agent') srvBadge = 'badge-langchain';

          const dur = e.duration_ms !== undefined ? `${e.duration_ms}ms` : '-';
          const stat = e.status_code ? `<span style="color:var(--emerald);">${e.status_code}</span>` : '-';
          const payloadStr = JSON.stringify(e.payload || {}, null, 2);

          tbody.innerHTML += `
            <tr>
              <td style="color:var(--text-subtle);">${(e.timestamp || '').split('T')[1] || e.timestamp}</td>
              <td><span class="badge-srv ${srvBadge}">${e.service || 'unknown'}</span></td>
              <td><span style="font-weight:600;">${e.type || 'event'}</span></td>
              <td>${e.endpoint || e.method || '-'}</td>
              <td>${stat} (${dur})</td>
              <td><pre class="json-preview">${escapeHtml(payloadStr)}</pre></td>
            </tr>
          `;
        });
      } catch (e) {
        console.error("Log load error:", e);
      }
    }

    function escapeHtml(str) {
      return (str || '').replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    // Initial load
    refreshStatus();
  </script>
</body>
</html>
"""


def start_web_app(port: int = 8000, host: str = "127.0.0.1"):
    """Starts the web app console server."""
    print("\n" + "=" * 62)
    print(f"  Starting Web App Console on http://{host}:{port}")
    print("=" * 62 + "\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    start_web_app(port=port)
