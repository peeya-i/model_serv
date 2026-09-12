import os
import sys
import warnings

# Suppress NVML initialization warnings caused by driver/library version mismatch
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# 1. Enforce air-gapped offline modes globally before importing frameworks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

# If HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE are set here, the program will only use downloaded models
# and will not be downloaded if it is missing
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Set the Hugging Face cache directory to a local path within the project
project_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(project_dir, "models")
os.environ["HF_HUB_CACHE"] = os.path.join(project_dir, "models")

import argparse
import uvloop
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.serve.utils.api_utils import cli_env_setup


if __name__ == "__main__":
    # Fast exit for help request
    if "-h" in sys.argv or "--help" in sys.argv:
        print("Usage: python serve_private_llm.py [-l <port>] [--model <model path or name>]")
        print("\nOptions:")
        print("  -l, --port <port>   Listening port for the model server (default: 8000)")
        print("  --model <model>     Model path or name (default: current local model)")
        sys.exit(0)

    cli_env_setup()

    cli_parser = argparse.ArgumentParser(add_help=False)
    cli_parser.add_argument(
        "-l", "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Listening port (default: 8000)"
    )
    cli_parser.add_argument(
        "--model",
        default=os.environ.get(
            "MODEL_PATH",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Llama-3.2-3B-Instruct")
        ),
        help="Model path or name (default: current local model)"
    )
    cli_args, _ = cli_parser.parse_known_args()
    listening_port = str(cli_args.port)
    served_model_name = os.path.basename(os.path.normpath(cli_args.model))

    # 2. Hardcode configuration args to prevent unauthorized command line overrides
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    
    # Define exact private runtime specifications
    custom_args = [
        "--model", cli_args.model,
        "--served-model-name", served_model_name, # Friendly model identifier for clients
        "--host", "127.0.0.1",                       # Bind to localhost or specific internal IP
        "--port", listening_port,                    # Configured listening port (default: 8000)
        "--api-key", "your-internal-secure-gateway-token-xyz", # Secure token authentication
        "--no-enable-log-requests",                   # Privacy setting: Never write prompts to logs
        "--enforce-eager",                           # Avoid CUDA graph memory overhead if needed
        "--gpu-memory-utilization", "0.85",          # Cap GPU utilization safely
        "--max-model-len", "2048",                   # Match TinyLlama's configured context window
        "--cpu-offload-gb", "3",                      # Offload ~3GB weights to system RAM to fit on 6GB GPU
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
