import os
import sys
import warnings

# Suppress NVML initialization warnings caused by driver/library version mismatch
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# 1. Load environment variables from .env file if present
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

env_file = os.path.join(project_dir, ".env")
if os.path.exists(env_file):
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

# Ensure Hugging Face authentication tokens are set
if os.environ.get("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])

# Set HF_HUB_OFFLINE="1" in the environment to enforce offline-only mode
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

# Set the Hugging Face cache directory to a local path within the project
os.environ["HF_HOME"] = os.path.join(project_dir, "models")
os.environ["HF_HUB_CACHE"] = os.path.join(project_dir, "models")

import argparse
import platform
import subprocess
import event_logger

def apply_turing_triton_patch():
    """Dynamically patches vLLM's Triton unified attention kernel for Turing (sm_75) GPUs.
    Turing GPUs have a 64KB shared memory hardware limit. For models running in float32
    with head_dim >= 256 (like Gemma 2), the default tile size of 32 requires 80KB shared memory,
    causing Triton to throw OutOfResources. Patching the tile size to 16 reduces shared memory
    to 48KB so it compiles and runs cleanly without modifying any library files in site-packages."""
    try:
        import vllm.v1.attention.ops.triton_unified_attention as tua
        _orig_get_tile_size = tua._get_tile_size

        def _turing_get_tile_size(head_size: int, sliding_window: int, element_size: int, is_prefill: bool) -> int:
            if is_prefill and element_size >= 4 and head_size >= 256:
                return 16
            return _orig_get_tile_size(head_size, sliding_window, element_size, is_prefill)

        tua._get_tile_size = _turing_get_tile_size
    except Exception as e:
        print(f"[Warning] Failed to apply Turing Triton patch: {e}", flush=True)

apply_turing_triton_patch()



def get_available_models(models_dir: str) -> list[tuple[str, str, int]]:
    """Discover local and cached models with their recommended max-model-len."""
    available = []

    # 1. Direct model directories containing config.json
    if os.path.isdir(models_dir):
        for entry in sorted(os.listdir(models_dir)):
            full_path = os.path.join(models_dir, entry)
            if os.path.isdir(full_path) and not entry.startswith(("models--", ".", "hub", "xet")):
                if os.path.exists(os.path.join(full_path, "config.json")):
                    max_len = 2048 if "tinyllama" in entry.lower() else 4096
                    available.append((f"models/{entry}", full_path, max_len))

    # 2. Hugging Face cache repositories
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir(models_dir)
        for repo in cache_info.repos:
            repo_id = repo.repo_id
            max_len = 2048 if "tinyllama" in repo_id.lower() else 4096
            if not any(item[0] == repo_id or item[1] == repo_id for item in available):
                available.append((repo_id, repo_id, max_len))
    except Exception:
        pass

    return available


def prompt_model_selection(models_dir: str) -> tuple[str, int]:
    """Prompt user to select an available model by number or enter custom parameters."""
    available = get_available_models(models_dir)

    print("\n" + "=" * 56, flush=True)
    print(" Select a model to serve:", flush=True)
    print("=" * 56, flush=True)
    for idx, (display_name, _, max_len) in enumerate(available, 1):
        print(f"  [{idx}] {display_name} (max-model-len: {max_len})", flush=True)
    custom_idx = len(available) + 1
    print(f"  [{custom_idx}] Enter model name and max-len manually", flush=True)
    print("=" * 56, flush=True)

    while True:
        try:
            choice = input(f"Select an option (1-{custom_idx}): ").strip()
            if not choice:
                continue
            choice_num = int(choice)
            if 1 <= choice_num <= len(available):
                _, model_target, max_len = available[choice_num - 1]
                return model_target, max_len
            elif choice_num == custom_idx:
                manual_model = input("Enter model name or path: ").strip().strip("'\"")
                if not manual_model:
                    print("Error: Model name cannot be empty.")
                    continue
                default_len = 2048 if "tinyllama" in manual_model.lower() else 4096
                manual_len_str = input(f"Enter max-model-len (default: {default_len}): ").strip().strip("'\"")
                try:
                    manual_len = int(manual_len_str) if manual_len_str else default_len
                except ValueError:
                    print(f"Invalid integer. Using default {default_len}.")
                    manual_len = default_len
                return manual_model, manual_len
            else:
                print(f"Please select a number between 1 and {custom_idx}.")
        except (ValueError, IndexError):
            print(f"Please enter a valid number (1-{custom_idx}).")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.")
            sys.exit(0)


_existing_instances_checked = False


def check_existing_instances(target_ports: list[int] = [8000, 8001, 8002]) -> list[dict]:
    """Scans the system for other running instances of the application or processes using its default ports."""
    current_pid = os.getpid()
    parent_pid = os.getppid() if hasattr(os, "getppid") else None
    detected = []
    seen_pids = set()

    app_keywords = ["serve_private_llm.py", "web_app.py", "vllm.entrypoints.openai", "vllm::enginecore", "enginecore"]

    # 1. Process search via psutil if installed
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                pid = proc.info["pid"]
                if pid in seen_pids or pid == current_pid or pid == parent_pid:
                    continue
                cmdline_list = proc.info.get("cmdline") or []
                cmdline = " ".join(cmdline_list)
                cmdline_lower = cmdline.lower()
                proc_name = (proc.info.get("name") or "").lower()
                if any(kw in cmdline_lower or kw in proc_name for kw in app_keywords):
                    seen_pids.add(pid)
                    detected.append({
                        "pid": pid,
                        "cmd": cmdline if cmdline else proc.info.get("name", "vllm"),
                        "name": proc.info.get("name", "python")
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        # Fallback via OS commands
        sys_type = platform.system()
        if sys_type in ["Linux", "Darwin"]:
            try:
                out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True, stderr=subprocess.DEVNULL)
                for line in out.splitlines():
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        pid = int(parts[0])
                        cmd = parts[1]
                        cmd_lower = cmd.lower()
                        if pid not in seen_pids and pid != current_pid and pid != parent_pid:
                            if any(kw in cmd_lower for kw in app_keywords):
                                seen_pids.add(pid)
                                detected.append({
                                    "pid": pid,
                                    "cmd": cmd,
                                    "name": "python"
                                })
            except Exception:
                pass
        elif sys_type == "Windows":
            try:
                out = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine"],
                    text=True,
                    stderr=subprocess.DEVNULL
                )
                for line in out.splitlines():
                    line_lower = line.lower()
                    if any(kw in line_lower for kw in app_keywords):
                        parts = line.strip().rsplit(None, 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            pid = int(parts[1])
                            if pid not in seen_pids and pid != current_pid and pid != parent_pid:
                                seen_pids.add(pid)
                                detected.append({
                                    "pid": pid,
                                    "cmd": parts[0],
                                    "name": "python"
                                })
            except Exception:
                pass

    # 2. Check for port-bound processes (handles background workers or standalone instances)
    if platform.system() in ["Linux", "Darwin"]:
        for port in target_ports:
            try:
                out = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True, stderr=subprocess.DEVNULL).strip()
                if out:
                    for p in out.split():
                        if p.isdigit():
                            pid = int(p)
                            if pid not in seen_pids and pid != current_pid and pid != parent_pid:
                                seen_pids.add(pid)
                                detected.append({
                                    "pid": pid,
                                    "cmd": f"Process listening on port {port}",
                                    "port": port
                                })
            except Exception:
                pass
    elif platform.system() == "Windows":
        for port in target_ports:
            try:
                out = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command", f"(Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue).OwningProcess"],
                    text=True,
                    stderr=subprocess.DEVNULL
                ).strip()
                if out:
                    for p in out.split():
                        if p.isdigit():
                            pid = int(p)
                            if pid not in seen_pids and pid != current_pid and pid != parent_pid:
                                seen_pids.add(pid)
                                detected.append({
                                    "pid": pid,
                                    "cmd": f"Process listening on port {port}",
                                    "port": port
                                })
            except Exception:
                pass

    return detected


def check_and_notify_existing_instances(target_ports: list[int] = [8000, 8001, 8002]):
    """Checks for other running instances of the app, notifies the user in the terminal with system-specific termination commands, and proceeds as normal."""
    global _existing_instances_checked
    if _existing_instances_checked:
        return
    _existing_instances_checked = True

    instances = check_existing_instances(target_ports)
    if not instances:
        return

    sys_type = platform.system()  # 'Linux', 'Darwin' (macOS), or 'Windows'
    pids = [str(inst["pid"]) for inst in instances]
    pids_space = " ".join(pids)
    pids_comma = ",".join(pids)

    ports_found = sorted({inst["port"] for inst in instances if inst.get("port")})

    print("\n" + "=" * 76, flush=True)
    print(" ⚠️  NOTICE: Another instance of this application is already running!", flush=True)
    print("=" * 76, flush=True)
    print(" Detected active instance(s):", flush=True)
    for inst in instances:
        port_info = f" [Port :{inst['port']}]" if inst.get("port") else ""
        cmd_text = inst["cmd"]
        if len(cmd_text) > 70:
            cmd_text = cmd_text[:67] + "..."
        print(f"   • PID {inst['pid']}{port_info} | {cmd_text}", flush=True)

    print("\n Command(s) you can use to terminate the existing instance(s):", flush=True)
    if sys_type == "Linux":
        print(f"   ► Graceful terminate by PID:   kill -15 {pids_space}", flush=True)
        print(f"   ► Force terminate by PID:      kill -9 {pids_space}", flush=True)
        print("   ► Terminate all by script:     pkill -f serve_private_llm.py", flush=True)
        for pt in ports_found:
            print(f"   ► Free port {pt}:               fuser -k {pt}/tcp", flush=True)
    elif sys_type == "Darwin":
        print(f"   ► Graceful terminate by PID:   kill -15 {pids_space}", flush=True)
        print(f"   ► Force terminate by PID:      kill -9 {pids_space}", flush=True)
        print("   ► Terminate all by script:     pkill -f serve_private_llm.py", flush=True)
        for pt in ports_found:
            print(f"   ► Free port {pt}:               lsof -ti :{pt} | xargs kill -9", flush=True)
    elif sys_type == "Windows":
        print(f"   ► Command Prompt (CMD):        taskkill /F /PID {pids[0]}", flush=True)
        print(f"   ► PowerShell:                  Stop-Process -Id {pids_comma} -Force", flush=True)
        print("   ► Stop all Python instances:   taskkill /F /IM python.exe", flush=True)
        for pt in ports_found:
            print(f"   ► Free port {pt} (PowerShell):   Get-Process -Id (Get-NetTCPConnection -LocalPort {pt}).OwningProcess | Stop-Process -Force", flush=True)
    else:
        print(f"   ► Terminate by PID:            kill -9 {pids_space}", flush=True)

    print("\n Proceeding as normal...", flush=True)
    print("=" * 76 + "\n", flush=True)


if __name__ == "__main__":
    # Fast exit for help request
    if "-h" in sys.argv or "--help" in sys.argv:
        print("Usage: python serve_private_llm.py [--cli] [-l <port>] [--model <model path or name>]")
        print("\nOptions:")
        print("  --cli               Run directly in terminal CLI mode (default port: 8001)")
        print("  -l, --port <port>   Listening port (default: 8000 for Web App, 8001 for --cli)")
        print("  --model <model>     Model path or name (default: interactive selection)")
        sys.exit(0)

    # Check and notify if another instance is already running
    check_and_notify_existing_instances(target_ports=[8000, 8001, 8002])

    cli_parser = argparse.ArgumentParser(add_help=False)
    cli_parser.add_argument(
        "-l", "--port",
        type=int,
        default=None,
        help="Listening port"
    )
    cli_parser.add_argument(
        "--model",
        default=None,
        help="Model path or name (default: current local model)"
    )
    cli_parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in terminal CLI mode directly"
    )
    cli_args, _ = cli_parser.parse_known_args()

    is_cli = cli_args.cli or any(arg == "--cli" for arg in sys.argv)

    # If started without the --cli flag, ask the user whether to start Web App or continue in terminal
    if not is_cli:
        print("\n" + "=" * 56, flush=True)
        print(" Select an option:", flush=True)
        print("=" * 56, flush=True)
        print("  [1] Start the Web App", flush=True)
        print("  [2] Continue Using the terminal", flush=True)
        print("=" * 56, flush=True)
        while True:
            try:
                choice = input("Select an option (1-2): ").strip()
                if choice in ["1", "web", "Start the Web App"]:
                    web_port = cli_args.port if cli_args.port is not None else int(os.environ.get("PORT", "8000"))
                    from web_app import start_web_app
                    start_web_app(port=web_port)
                    sys.exit(0)
                elif choice in ["2", "terminal", "Continue Using the terminal"]:
                    break
                else:
                    print("Please select 1 or 2.", flush=True)
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled by user.", flush=True)
                sys.exit(0)

    models_dir = os.path.join(project_dir, "models")
    has_model_arg = any(arg == "--model" or arg.startswith("--model=") for arg in sys.argv)

    # When serving the model server via terminal/CLI, default to port 8001 unless overridden
    default_port = 8001
    if cli_args.port is not None:
        listening_port = str(cli_args.port)
    else:
        listening_port = str(os.environ.get("PORT", str(default_port)))

    # If --model is not provided via CLI, prompt the user interactively from models folder
    if has_model_arg and cli_args.model:
        selected_model = cli_args.model.strip().strip("'\"")
        if "tinyllama" in selected_model.lower():
            max_model_len = "2048"
        else:
            max_model_len = "4096"
    else:
        selected_model, chosen_len = prompt_model_selection(models_dir)
        selected_model = selected_model.strip().strip("'\"")
        max_model_len = str(chosen_len)

    served_model_name = os.path.basename(os.path.normpath(selected_model))
    served_model_names = [served_model_name]
    if selected_model != served_model_name:
        served_model_names.append(selected_model)

    print(f"\n[Info] Launching server for '{served_model_name}' on port {listening_port} (max-model-len: {max_model_len})...\n")

    # 2. Hardcode configuration args to prevent unauthorized command line overrides
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
    import uvloop

    cli_env_setup()

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)


    # Dynamic cpu-offload-gb based on model size:
    # TinyLlama (1.1B) is only ~2.05 GB and fits entirely inside 6GB GPU VRAM with plenty of KV cache room.
    # Forcing CPU offload on 1B models triggers UVAOffloader over PCIe, slowing token generation down to ~2 tokens/sec.
    is_gemma = "gemma" in selected_model.lower()
    needs_cpu_offload = "tinyllama" not in selected_model.lower()

    # Gemma 2 rejects float16 due to soft-capping instability and falls back to float32 on Turing (sm_75).
    # In float32 it requires ~10.4GB, so we offload 7.5GB to CPU RAM. For other models, float16 is used.
    model_dtype = "auto" if is_gemma else "float16"

    if is_gemma:
        max_model_len = "1536"

    gpu_mem_util = "0.72" if is_gemma else "0.85"

    # Define exact private runtime specifications
    custom_args = [
        "--model", selected_model,
        "--served-model-name", *served_model_names, # Friendly model identifiers for clients
        "--host", "127.0.0.1",                       # Bind to localhost or specific internal IP
        "--port", listening_port,                    # Configured listening port (default: 8000)
        "--api-key", "your-internal-secure-gateway-token-xyz", # Secure token authentication
        "--middleware", "event_logger.vllm_logging_middleware", # Complete request/response event logging
        "--no-enable-log-requests",                   # Privacy setting: Never write prompts to logs
        "--enforce-eager",                           # Avoid CUDA graph memory overhead if needed
        "--gpu-memory-utilization", gpu_mem_util,    # Cap GPU utilization safely with headroom for activations
        "--max-model-len", max_model_len,            # Set dynamically based on model loaded
        "--attention-backend", "TRITON_ATTN",         # Explicitly use Triton attention for Turing (sm_75) GPUs
        "--dtype", model_dtype,
    ]
    if is_gemma:
        custom_args.extend(["--cpu-offload-gb", "7.5"])
    elif needs_cpu_offload:
        custom_args.extend(["--cpu-offload-gb", "3"])

    # Only Llama 3.2 tokenizers provide the special tokens required by this parser.
    if "llama-3.2" in served_model_name.lower():
        custom_args.extend([
            "--enable-auto-tool-choice",
            "--tool-call-parser", "llama3_json",
        ])
    
    args = parser.parse_args(custom_args)
    validate_parsed_serve_args(args)

    # 3. Record model startup time for telemetry
    try:
        import time
        start_time_file = os.path.join(project_dir, "logs", "model_start_time.txt")
        os.makedirs(os.path.dirname(start_time_file), exist_ok=True)
        with open(start_time_file, "w", encoding="utf-8") as f:
            f.write(str(time.time()))
    except Exception:
        pass

    # Initialize and run the high-performance async inference engine
    uvloop.run(run_server(args))
