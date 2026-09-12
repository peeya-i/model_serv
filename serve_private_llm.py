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
import uvloop
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
import event_logger


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

    print("\n" + "=" * 56)
    print(" Select a model to serve:")
    print("=" * 56)
    for idx, (display_name, _, max_len) in enumerate(available, 1):
        print(f"  [{idx}] {display_name} (max-model-len: {max_len})")
    custom_idx = len(available) + 1
    print(f"  [{custom_idx}] Enter model name and max-len manually")
    print("=" * 56)

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


if __name__ == "__main__":
    # Fast exit for help request
    if "-h" in sys.argv or "--help" in sys.argv:
        print("Usage: python serve_private_llm.py [-l <port>] [--model <model path or name>]")
        print("\nOptions:")
        print("  -l, --port <port>   Listening port for the model server (default: 8000)")
        print("  --model <model>     Model path or name (default: interactive selection)")
        sys.exit(0)

    cli_env_setup()

    models_dir = os.path.join(project_dir, "models")
    has_model_arg = any(arg == "--model" or arg.startswith("--model=") for arg in sys.argv)

    cli_parser = argparse.ArgumentParser(add_help=False)
    cli_parser.add_argument(
        "-l", "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Listening port (default: 8000)"
    )
    cli_parser.add_argument(
        "--model",
        default=None,
        help="Model path or name (default: current local model)"
    )
    cli_args, _ = cli_parser.parse_known_args()
    listening_port = str(cli_args.port)

    # If --model is not provided via CLI, prompt the user interactively
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
    print(f"\n[Info] Launching server for '{served_model_name}' on port {listening_port} (max-model-len: {max_model_len})...\n")

    # 2. Hardcode configuration args to prevent unauthorized command line overrides
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)

    # Define exact private runtime specifications
    custom_args = [
        "--model", selected_model,
        "--served-model-name", served_model_name, # Friendly model identifier for clients
        "--host", "127.0.0.1",                       # Bind to localhost or specific internal IP
        "--port", listening_port,                    # Configured listening port (default: 8000)
        "--api-key", "your-internal-secure-gateway-token-xyz", # Secure token authentication
        "--middleware", "event_logger.vllm_logging_middleware", # Complete request/response event logging
        "--no-enable-log-requests",                   # Privacy setting: Never write prompts to logs
        "--enforce-eager",                           # Avoid CUDA graph memory overhead if needed
        "--gpu-memory-utilization", "0.85",          # Cap GPU utilization safely
        "--max-model-len", max_model_len,            # Set dynamically based on model loaded
        "--cpu-offload-gb", "3",                      # Offload ~3GB weights to system RAM to fit on 6GB GPU
        "--attention-backend", "TRITON_ATTN",         # Explicitly use Triton attention for Turing (sm_75) GPUs
    ]

    # Only Llama 3.2 tokenizers provide the special tokens required by this parser.
    if "llama-3.2" in served_model_name.lower():
        custom_args.extend([
            "--enable-auto-tool-choice",
            "--tool-call-parser", "llama3_json",
        ])
    
    args = parser.parse_args(custom_args)
    validate_parsed_serve_args(args)
    
    # 3. Initialize and run the high-performance async inference engine
    uvloop.run(run_server(args))
