# Private LLM & Local Agent Server (`model_serv`)

An air-gapped, high-performance local AI inference and agent server powered by **vLLM** and Meta's **Llama-3.2-3B-Instruct**.

---

## 🏛 System Architecture

This repository supports two interaction models:
1. **Direct Mode (Without Agent)**: Applications query the local OpenAI-compatible inference engine directly on port `8000` (ideal for private RAG pipelines, text generation, and direct prompt completion).
2. **Agent Mode (Server Agent Front)**: An intelligent intermediary layer on port `8001` (or via LangChain) that inspects user queries, automatically selects and executes local tools (calculators, system monitors, custom Python functions), and synthesizes tool outputs before returning answers to the user.

```
                  ┌─────────────────────────────────────────────────────────────┐
                  │               Client / Frontend / UI / User                 │
                  └──────────────┬───────────────────────────────┬──────────────┘
                                 │                               │
        [Option A: Direct Mode]  │                               │ [Option B: Agent Mode]
        OpenAI REST Requests     │                               │ HTTP REST Requests
        (No Tool Execution)      │                               │ (Tools + Orchestration)
                                 │                               ▼
                                 │                ┌──────────────────────────────┐
                                 │                │    Local Agent Server Front  │
                                 │                │(agent/agent_server.py :8001) │
                                 │                │   - Tool Registry            │
                                 │                │   - Execution & Reason Loop  │
                                 │                └──────────────┬───────────────┘
                                 │                               │
                                 ▼                               ▼
                 ┌───────────────────────────────────────────────────────────────┐
                 │       Local vLLM Model Server (serve_private_llm.py)          │
                 │                    Port 8000 (`/v1`)                          │
                 │         Model: Meta Llama-3.2-3B-Instruct                     │
                 │  - Air-gapped offline operation (Zero telemetry/external calls)│
                 │  - Zero prompt logging (High privacy & confidentiality)       │
                 │  - Hardware-optimized: 3GB CPU offload for 6GB VRAM GPUs      │
                 │  - Tool-calling parser enabled (llama3_json)                  │
                 └───────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Server Configuration & Specifications

| Setting | Value | Notes |
|---|---|---|
| **Model Engine** | `vLLM 0.28.0` | High-throughput async PagedAttention engine |
| **Model Name** | `Llama-3.2-3B-Instruct` | Local weights in `./models/Llama-3.2-3B-Instruct` |
| **Model Server URL** | `http://127.0.0.1:8000/v1` | OpenAI-compatible REST API |
| **Agent Server URL** | `http://127.0.0.1:8001` | FastAPI tool-augmented agent microservice |
| **API Key** | `your-internal-secure-gateway-token-xyz` | Token bearer authentication |
| **Context Window** | `4096` tokens | Accommodates tool schemas, conversation history, and RAG chunks |
| **Memory Allocation** | 0.85 GPU Util + 3GB RAM Offload | Tuned for ~6GB VRAM (Quadro RTX 3000 / RTX 2060 / GTX 1660) |
| **Tool Calling Parser** | `llama3_json` | Enables structured function calling for agents |

---

## 🛠️ Step 0: Setting up the Platform

### 🔑 1. Hugging Face Prerequisites (All Operating Systems)
1. **Request Model Access**: Visit [huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and accept the community license terms. Access is typically granted within minutes.
2. **Generate Access Token**:
   - In Hugging Face, click your profile icon (top-right) -> **Access Tokens**.
   - Click **+ Create new token** (type: Read).
   - Copy the token; you will need it in the configuration step below.

---

### 💻 2. System-Specific Installation & Setup

Choose your operating system below to set up your directory, Python environment, dependencies, and GPU drivers:

#### 🐧 Option A: Linux (Ubuntu / Debian)

1. **Terminal & Project Directory**:
   ```bash
   mkdir -p ~/Documents/model_serv
   cd ~/Documents/model_serv
   ```

2. **Python Virtual Environment**:
   ```bash
   sudo apt update && sudo apt install -y python3-venv python3-pip
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -U transformers torch==2.13.0 "huggingface_hub[cli]" accelerate
   ```

4. **Install & Verify NVIDIA GPU Drivers**:
   ```bash
   # If nvidia-smi is not yet installed:
   sudo apt install -y nvidia-utils-550  # or: sudo ubuntu-drivers install

   # Verify GPU name, VRAM, and driver version:
   nvidia-smi

   # Query specific properties in clean tabular format:
   nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv
   ```

---

#### 🪟 Option B: Windows PC (Native & WSL2)

1. **PowerShell & Project Directory**:
   ```powershell
   mkdir $HOME\Documents\model_serv
   cd $HOME\Documents\model_serv
   ```

2. **Python Virtual Environment**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   # Note: If script execution is restricted on Windows, run once:
   # Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   *(Or in Command Prompt: `.venv\Scripts\activate.bat`)*

3. **Install Dependencies**:
   ```powershell
   python -m pip install --upgrade pip
   pip install -U transformers torch==2.13.0 "huggingface_hub[cli]" accelerate
   ```

4. **Install & Verify NVIDIA GPU Drivers**:
   - **Native Windows**:
     1. Download and install the latest NVIDIA Driver from [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx) (or via GeForce Experience / NVIDIA App).
     2. The installer automatically provides `nvidia-smi.exe` and registers it in `C:\Windows\System32\`.
     3. Verify in PowerShell or Command Prompt:
        ```powershell
        nvidia-smi
        nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv
        ```
     *(GUI Alternative: Press `Ctrl + Shift + Esc` -> select **Performance** -> **GPU**, or run `dxdiag`)*
   - **WSL2 (Windows Subsystem for Linux)**:
     Install the NVIDIA Windows driver on the host machine. WSL2 automatically inherits GPU acceleration and CUDA support; run `nvidia-smi` directly in your WSL2 terminal.

---

#### 🍏 Option C: macOS (MacBook — Apple Silicon M-Series & Intel)

1. **Terminal & Project Directory**:
   ```bash
   mkdir -p ~/Documents/model_serv
   cd ~/Documents/model_serv
   ```

2. **Python Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -U transformers torch "huggingface_hub[cli]" accelerate
   ```

4. **Inspect Chip, GPU Cores & Unified Memory**:
   > **Note on Apple Silicon**: MacBooks with M1, M2, M3, or M4 chips utilize an SoC architecture with high-bandwidth Unified Memory shared across CPU and GPU cores rather than a discrete NVIDIA GPU.

   ```bash
   # View chip model, CPU/GPU core count, and Unified Memory:
   system_profiler SPHardwareDataType SPDisplaysDataType | grep -E "Chip|Memory|Cores"

   # Real-time GPU power and frequency usage (built-in):
   sudo powermetrics --samplers gpu_power -i 1000 -n 1

   # Optional terminal GPU monitor (similar to nvidia-smi / htop):
   pip install asitop && asitop

   # Verify PyTorch Apple Metal (MPS) GPU acceleration:
   python3 -c "import torch; print('Apple GPU (MPS) Available:', torch.backends.mps.is_available())"
   ```
   *(GUI Alternative: Click Apple Menu `` -> **About This Mac** -> **System Report...** -> **Graphics/Displays**)*

---

### ⚙️ 3. Project Configuration & Model Storage (All Operating Systems)

1. **Authenticate with Hugging Face (`.env`)**:
   Create a `.env` file in the project root:
   ```bash
   echo "HF_TOKEN=hf_your_actual_token_here" > .env
   ```
   *(On Windows PowerShell: `Set-Content -Path .env -Value "HF_TOKEN=hf_your_actual_token_here"`)*
   
   *`serve_private_llm.py` automatically reads `HF_TOKEN` from `.env` to authenticate model downloads.*

2. **Create Models Directory**:
   ```bash
   mkdir -p models
   ```
   *(Optional manual pre-download, or let `serve_private_llm.py` download automatically on first run)*:
   ```bash
   hf download meta-llama/Llama-3.2-3B-Instruct --local-dir ./models/Llama-3.2-3B-Instruct
   ```


## 🚀 Step 1: Start the Local Model Server

Before running direct queries or starting an agent, launch the core inference server.

### Virtual Environment (Recommended)

#### Command Options & Parameters

| Parameter | Shorthand | Description | Default |
| :--- | :--- | :--- | :--- |
| `--model <model>` | — | Model path or Hugging Face model identifier | Interactive Menu |
| `--port <port>` | `-l <port>` | Listening port for the model server | `8000` |

#### Usage Examples

**1. Interactive Selection & Automatic Model Loading (when started without `--model`):**
```bash
cd /home/<uname>/Documents/model_serv
./.venv/bin/python serve_private_llm.py
```
This displays an interactive menu of local models and allows typing a new model:
```text
========================================================
 Select a model to serve:
========================================================
  [1] models/Llama-3.2-3B-Instruct (max-model-len: 4096)
  [2] TinyLlama/TinyLlama-1.1B-Chat-v1.0 (max-model-len: 2048)
  [3] Enter model name and max-len manually
========================================================
Select an option (1-3): 
```

- **Select a local model**: Enter `1` or `2`.
- **Automatically download and load a new model**: Enter `3`, then type the Hugging Face repository name. If the model is not already cached, it will be **automatically downloaded and loaded** using the `HF_TOKEN` from your `.env` file!

##### Examples of Valid Model Names:
| Model Type | Valid Model Name / Identifier | Suggested `max-model-len` | Notes |
| :--- | :--- | :--- | :--- |
| **Qwen 2.5 Coder** | `Qwen/Qwen2.5-Coder-3B-Instruct` | `4096` | High performance coding model |
| **Qwen 2.5 Small** | `Qwen/Qwen2.5-1.5B-Instruct` | `4096` | Lightweight & fast |
| **TinyLlama** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `2048` | Compact, 2048 max context |
| **Llama 3.2** | `meta-llama/Llama-3.2-3B-Instruct` | `4096` | Requires HF gated repo access |
| **Phi 3.5** | `microsoft/Phi-3.5-mini-instruct` | `4096` | High-accuracy 3.8B model |

> [!NOTE]
> **GPU Compatibility (Turing sm_75)**: Check the GPU on the machine and make sure it will be able to support the model being loaded. For instance, the Quadro RTX 3000 supports `float16` and `float32`, but does not have native hardware tensor cores for `bfloat16`. Models like `Llama 3.2`, `Qwen 2.5`, and `TinyLlama` support `float16` casting cleanly. However, models like `Gemma 2` forbid `float16` due to numerical instability and force `float32` upcasting, which doubles memory consumption to ~10.5 GB and exceeds 6GB VRAM.

**2. Custom Listening Port (using `-l` or `--port`):**
```bash
./.venv/bin/python serve_private_llm.py -l 8500
```

**3. Direct CLI Model Launch (Automatic Download if not cached):**
```bash
./.venv/bin/python serve_private_llm.py --model "Qwen/Qwen2.5-Coder-3B-Instruct"
```

**4. Custom Model and Port Combined:**
```bash
./.venv/bin/python serve_private_llm.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" -l 8500
```

*The model server initializes weights, offloads 3GB to system RAM, sets up Triton attention, and listens on `http://127.0.0.1:<port>/v1` (default: 8000).*

---

## 💬 Step 2: Use the Model Directly (Without Server Agent)

In this mode, downstream systems (e.g. private RAG, standard chat interfaces) send prompts directly to the model server without any autonomous tool calling.

### 1. Run Verification Test Script

Open a separate terminal and run:

```bash
./.venv/bin/python test_server.py
```

**Expected Output:**
```text
Checking GET http://127.0.0.1:8000/v1/models ...
Status Code: 200
Response: {'data': [{'id': 'Llama-3.2-3B-Instruct', 'max_model_len': 4096, ...}]}

Sending test prompt to POST http://127.0.0.1:8000/v1/chat/completions ...
Status Code: 200
Generated Message: Private LLM server is up and running!
```

### 2. Direct Query via `curl`

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-internal-secure-gateway-token-xyz" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain what Retrieval-Augmented Generation (RAG) is in two sentences."}
    ],
    "max_tokens": 100,
    "temperature": 0.2
  }'
```

### 3. Direct Query in Python (OpenAI SDK / Private RAG)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="your-internal-secure-gateway-token-xyz"
)

response = client.chat.completions.create(
    model="Llama-3.2-3B-Instruct",
    messages=[
        {"role": "system", "content": "You are a private technical assistant."},
        {"role": "user", "content": "Summarize the key benefits of local model inference."}
    ],
    temperature=0.2,
    max_tokens=256
)

print(response.choices[0].message.content)
```

---

## 🤖 Step 3: Start and Use the Server Agent as a Front for the Model

When you need the model to autonomously reason, decide when external data or actions are required, and execute tools, run one of the two server agent options.

### Option 1: FastAPI Microservice Front (`agent/agent_server.py`)

A persistent HTTP service listening on port `8001`. It receives user prompts, passes them to the local model alongside registered tool schemas, intercepts function calls, executes the local Python functions, and returns the synthesized answer.

#### Available Tools
- `calculate(expression)`: Mathematical calculations.
- `lookup_system_status()`: Inspects hardware and system metrics.

#### 1. Start the Agent Server
In a separate terminal:

**Default ports (listen: 8001, model: 8000):**
```bash
./.venv/bin/python agent/agent_server.py
```

**Custom ports (specify listening port `-l` and/or model server port `-m`):**
```bash
# Listen on 8001, connect to model server on custom port 8500
./.venv/bin/python agent/agent_server.py -m 8500

# Listen on custom port 9001, connect to model server on default port 8000
./.venv/bin/python agent/agent_server.py -l 9001

# Specify both listening and model ports
./.venv/bin/python agent/agent_server.py -l 9001 -m 8500
```

#### 2. Check Agent Health
```bash
curl -s http://127.0.0.1:8001/health
```

**Response:**
```json
{"status":"healthy","service":"local_agent_server","model":"Llama-3.2-3B-Instruct"}
```

#### 3. Send a Tool-Requiring Prompt to the Agent Front
```bash
curl -X POST http://127.0.0.1:8001/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 45 * 20?"}'
```

**Response:**
```json
{
  "reply": "The result of 45 * 20 is 900.",
  "tools_used": ["calculate"]
}
```

---

### Option 2: LangChain Python Agent (`agent/langchain_agent.py`)

A Python-native agent using `langchain_core` and `langchain_openai`. Ideal for integrating directly into larger Python backends, background workflows, or automation pipelines.

#### Available Tools
- `get_current_time()`: Retrieves formatted system timestamp.
- `calculate(expression)`: Computes arithmetic expressions.

#### 1. Run the Agent CLI Test
```bash
./.venv/bin/python agent/langchain_agent.py
```

**Output:**
```text
=== Testing LangChain Local Server Agent ===

[User Query]: What time is it right now?
[Tool Call]: get_current_time with arguments: {}
[Tool Output]: 2026-09-10 16:02:56
[Agent Response]: The current time is 4:02 PM.

[User Query]: Calculate 125 * 8
[Tool Call]: calculate with arguments: {'expression': '125 * 8'}
[Tool Output]: 1000
[Agent Response]: The result of 125 * 8 is 1000.
```

#### 2. Import into Your Applications
```python
from agent.langchain_agent import run_agent

answer = run_agent("What time is it right now and what is 250 * 4?")
print(answer)
```

---

## 📊 Agent Options Comparison

| Feature | Direct Mode (`:8000`) | Option 1: `agent/agent_server.py` (`:8001`) | Option 2: `agent/langchain_agent.py` |
|---|---|---|---|
| **Role** | Raw LLM Inference Engine | Front-end Agent Microservice | Python Agent Pipeline |
| **Tools / Skills** | None (Raw Completion) | `calculate`, `lookup_system_status` | `get_current_time`, `calculate` |
| **Interface** | OpenAI REST API (`/v1`) | REST API (`/agent/chat`, `/health`) | Python Function / CLI |
| **Protocol** | HTTP on `:8000` | HTTP on `:8001` | In-process Python invocation |
| **Best For** | Private RAG, embeddings, direct completions | Web frontends, mobile apps, microservices | Python workflows, notebooks, pipelines |

---

## 🛑 Step 4: Shut Down

### 1. Stopping the Agent Server (`agent/agent_server.py`)
- If running in foreground: Press `Ctrl + C`
- If running in background:
  - **Linux / macOS**:
    ```bash
    kill $(lsof -t -i:8001)
    ```
  - **Windows (PowerShell)**:
    ```powershell
    Stop-Process -Id (Get-NetTCPConnection -LocalPort 8001).OwningProcess -Force
    ```

### 2. Stopping the Model Server (`serve_private_llm.py`)
- If running in foreground: Press `Ctrl + C`
- If running in background:
  - **Linux / macOS**:
    ```bash
    pkill -f "serve_private_llm.py"
    # Or: kill $(lsof -t -i:8000)
    ```
  - **Windows (PowerShell)**:
    ```powershell
    Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force
    ```


---

## 🛠 Technical Notes & Troubleshooting

- **Llama 3.2 Single-Tool-Call Template Rule**:
  Meta's official chat template for `Llama-3.2-3B-Instruct` enforces that each assistant turn contains at most one tool call (`"This model only supports single tool-calls at once!"`). Both `agent/agent_server.py` and `agent/langchain_agent.py` are structured to execute tool calls sequentially to ensure strict compatibility.
- **Offline / Air-Gapped Operation**:
  The flags `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` ensure no connection attempts are made to the public internet.
- **Privacy & Prompt Logging**:
  `--no-enable-log-requests` guarantees that sensitive prompts, retrieved documents, and tool payloads remain in volatile memory and are never written to disk logs.
