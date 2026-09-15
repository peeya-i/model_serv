# Private LLM & Local Agent Server (`model_serv`)

An air-gapped, high-performance local AI inference and agent server powered by **vLLM** and Meta's **Llama-3.2-3B-Instruct**, featuring a real-time Web Dashboard, server-side agent execution with tool calling, and FIFO-capped event logging.

---

## 🏛 System Architecture

This repository provides three primary ways to interact with your local models and agents:
1. **Interactive Web App Dashboard (Port 8002)**: A full-featured web console featuring server-side agent management with mutual exclusion, model selection dropdown with custom text input, real-time GPU hardware inspection, an interactive chat playground, and a live event log viewer.
2. **Direct Mode (Without Agent - Port 8000)**: Applications query the local OpenAI-compatible vLLM inference engine directly on port `8000` (ideal for private RAG pipelines, text generation, and direct prompt completion).
3. **Agent Mode (Server Agent Front - Port 8001)**: An intelligent microservice layer in `agents/` that inspects user queries, automatically selects and executes local tools (calculators, system status diagnostics, timestamps), and synthesizes tool outputs before returning answers.

```
                  ┌─────────────────────────────────────────────────────────────┐
                  │               Client / Frontend / UI / User                 │
                  └──────────────┬───────────────────────────────┬──────────────┘
                                 │                               │
        [Option A: Direct Mode]  │                               │ [Option B: Agent Mode]
        OpenAI REST Requests     │                               │ HTTP REST Requests
        (Port 8000 /v1)          │                               │ (Port 8001 /agent/chat)
                                 │                               ▼
                                 │                ┌──────────────────────────────┐
                                 │                │    Local Agent Server Front  │
                                 │                │ (agents/agent_server.py:8001)│
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
                 │  - Structured logging to logs/events.json (100MB FIFO capped) │
                 │  - Hardware-optimized: 3GB CPU offload for 6GB VRAM GPUs      │
                 │  - Tool-calling parser enabled (llama3_json)                  │
                 └───────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Server Configuration & Specifications

| Setting | Value | Notes |
|---|---|---|
| **Web App Dashboard** | `http://127.0.0.1:8002` | 3-tab web console for agents, models, chat, logs, and GPU metrics |
| **Model Server URL** | `http://127.0.0.1:8000/v1` | OpenAI-compatible REST API powered by vLLM |
| **Agent Server URL** | `http://127.0.0.1:8001` | FastAPI tool-augmented agent microservice (`agents/agent_server.py`) |
| **Model Engine** | `vLLM 0.28.0` | High-throughput async PagedAttention engine |
| **Default Model** | `Llama-3.2-3B-Instruct` | Local weights in `./models/Llama-3.2-3B-Instruct` |
| **API Key** | `your-internal-secure-gateway-token-xyz` | Token bearer authentication |
| **Context Window** | `4096` tokens | Accommodates tool schemas, conversation history, and RAG chunks |
| **Memory Allocation** | 0.85 GPU Util + 3GB RAM Offload | Tuned for ~6GB VRAM (Quadro RTX 3000 / RTX 2060 / GTX 1660) |
| **Tool Calling Parser** | `llama3_json` | Enables structured function calling for agents |
| **Event Logger** | `logs/events.json` | Complete payload logging with 100MB max FIFO rotation |

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


## 🚀 Step 1: Launch the Model Server & Web App

You can launch the system using either the interactive **Web App Console** or directly via the **Terminal CLI**.

### Launch Options

#### 1. Interactive Launch (Default)
Running without arguments prompts you to select between the **Web App Console** or the **Terminal CLI**:
```bash
./.venv/bin/python serve_private_llm.py
```
```text
========================================================
 Select an option:
========================================================
  [1] Start the Web App
  [2] Continue Using the terminal
========================================================
Select an option (1-2): 
```

##### Option [1]: Web App Console (`http://127.0.0.1:8002`)
Launches the full 3-page web dashboard on port `8002` (or specified `-l <port>`):
- **Tab 1: Server & Agents**:
  - **Server-Side Agent Selection (Dropdown)**:
    - Lists all agent microservice scripts discovered in `agents/` (`agent_server.py`, `langchain_agent.py`).
    - **Single Running Agent Rule (Mutual Exclusion)**: Only 1 server agent can be running at a time. Starting or switching to a new agent automatically terminates any previously running agent cleanly.
    - Configurable agent listening port (default: `8001`).
    - Dynamic action button: **"Enable Agent"** (Emerald) when stopped, **"Disable Agent"** (Rose) when currently running, or **"Switch to this Agent"** (Indigo) when a different agent is selected while one is active.
    - Live status badge showing active agent filename, port, and process ID (PID).
  - **Local Model Inference Engine (Dropdown + Text Box)**:
    - **Dropdown choice**: Lists discovered local weights in `models/` (e.g. `Llama-3.2-3B-Instruct`) and Hugging Face cached repositories, plus a `Custom Model (Enter in text box)...` option.
    - **Editable Text Box**: Shows the chosen model path and allows typing/editing any custom local directory path or remote Hugging Face repository (with bidirectional dropdown synchronization).
    - Configurable model server listening port (default: `8000`).
    - **Real-Time Active Readiness Verification**: Rather than falsely reporting "running" the moment a process spawns, the backend performs live HTTP readiness checks against `http://127.0.0.1:<port>/health` and `/v1/models`.
    - **Distinct Lifecycle States & Indicators**:
      - ⚪ **Stopped**: Gray badge (`Server Stopped`), action button: **"▶ Enable Model"**.
      - ⏳ **Loading / Initializing**: Amber pulsing badge (`⏳ Initializing & Loading Weights (~14s)...`), action button: **"⏳ Starting... (Click to Abort)"**, detailed status noting weight loading into memory and KV cache warmup. Frontend dynamically polls every 1.5s for immediate responsiveness.
      - 🟢 **Actively Running & Ready**: Vibrant emerald green pulsing badge (`● Actively Serving: <model_name> (:8000, PID: ...)`), action button: **"⏹ Stop Model Server"**, and live REST endpoint indicator.
      - ❌ **Crashed / Terminated**: Rose red badge (`Process Terminated (Exit: ...)`) with error log snippet preview.
    - **Persistent Top Navigation Pill**: A dedicated status indicator in the top navbar (`LLM: Actively Serving` / `LLM: Loading` / `LLM: Stopped`) provides immediate visibility across all 3 tabs.
  - **GPU Hardware & Compute Capabilities Panel** (prominently displayed below model selection):
    - **GPU Model Name / Number**: Real-time identification (e.g., `Quadro RTX 3000 with Max-Q Design`) with active driver version and CUDA runtime level.
    - **Architecture & Capabilities**: Hardware compute capability architecture (e.g., `Compute Capability 7.5 (Turing) | FP16/FP32 Acceleration`).
    - **Number of Processors**: Streaming Multiprocessors (SMs) and estimated parallel CUDA cores alongside host CPU threads.
    - **Number of GPUs**: Detected PCIe accelerator count.
    - **RAM Available on GPU (VRAM)**: Real-time free vs total VRAM metrics with visual animated meter bar and 3GB system RAM CPU offload indication.
- **Tab 2: Chat Console**:
  - Interactive playground allowing prompt execution against either the active **Server Agent** (port `8001`) or directly to the **Model Server** (port `8000`).
  - Target readiness badge displays live availability (`Direct Model Ready (:8000)` vs `Model Loading Weights...` vs `Model Offline`).
  - Prevents querying unready models with clear, helpful alerts if the server is still warming up.
  - Automatically displays executed tools and arguments when querying via the agent front.
- **Tab 3: Event Logs**:
  - Real-time tabular viewer for `logs/events.json`. Displays timestamps, service badges (`agent_server`, `langchain_agent`, `model_server`), route endpoints, HTTP status codes, durations, and complete expandable JSON request/response payloads.

##### Option [2]: Terminal CLI
Continues with interactive terminal model selection from the `models/` directory, defaulting to port `8000`.

---

#### 2. Direct CLI Launch (`--cli`)
Passing `--cli` skips the interactive prompt, defaults to port `8000`, and directly opens the terminal model selection menu:
```bash
./.venv/bin/python serve_private_llm.py --cli
```

### CLI Command Options & Parameters

| Parameter | Shorthand | Description | Default |
| :--- | :--- | :--- | :--- |
| `--cli` | — | Run directly in terminal CLI mode | Interactive Choice |
| `--model <model>` | — | Model path or Hugging Face model identifier | Interactive Menu |
| `--port <port>` | `-l <port>` | Listening port | `8002` (Web App) / `8000` (`--cli`) |

#### Usage Examples

**1. Interactive Selection in Terminal (when started without `--model`):**
```bash
./.venv/bin/python serve_private_llm.py --cli
```
Displays discovered models and option for manual entry:
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

**2. Custom Listening Port (using `-l` or `--port`):**
```bash
./.venv/bin/python serve_private_llm.py --cli -l 8500
```

**3. Direct CLI Model Launch (Automatic Download if not cached):**
```bash
./.venv/bin/python serve_private_llm.py --cli --model "Qwen/Qwen2.5-Coder-3B-Instruct"
```

**4. Custom Model and Port Combined:**
```bash
./.venv/bin/python serve_private_llm.py --cli --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" -l 8500
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
*(Optionally pass a custom port, e.g. `./.venv/bin/python test_server.py 8500`)*

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

When you need the model to autonomously reason, decide when external data or actions are required, and execute tools, run one of the server agents in `agents/`.

> **Note**: In the Web App Console, agent selection is a **single dropdown** with **strict mutual exclusion**: only 1 agent runs at a time. Starting an agent terminates any other running agent automatically.

### Option 1: FastAPI Microservice Front (`agents/agent_server.py`)

A persistent HTTP service listening on port `8001`. It receives user prompts, passes them to the local model alongside registered tool schemas, intercepts function calls, executes the local Python functions, and returns the synthesized answer.

#### Available Tools
- `calculate(expression)`: Mathematical calculations.
- `lookup_system_status()`: Inspects hardware and system metrics.

#### 1. Start the Agent Server
In a separate terminal (or launch directly from Web App Tab 1):

**Default ports (listen: 8001, model: 8000):**
```bash
./.venv/bin/python agents/agent_server.py
```

**Custom ports (specify listening port `-l` and/or model server port `-m`):**
```bash
# Listen on 8001, connect to model server on custom port 8500
./.venv/bin/python agents/agent_server.py -m 8500

# Listen on custom port 9002, connect to model server on default port 8000
./.venv/bin/python agents/agent_server.py -l 9002

# Specify both listening and model ports
./.venv/bin/python agents/agent_server.py -l 9002 -m 8500
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

### Option 2: LangChain Python Agent (`agents/langchain_agent.py`)

A Python-native agent using `langchain_core` and `langchain_openai`. Ideal for integrating directly into larger Python backends, background workflows, or automation pipelines.

#### Available Tools
- `get_current_time()`: Retrieves formatted system timestamp.
- `calculate(expression)`: Computes arithmetic expressions.

#### 1. Run the Agent CLI Test
```bash
./.venv/bin/python agents/langchain_agent.py
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
from agents.langchain_agent import run_agent

answer = run_agent("What time is it right now and what is 250 * 4?")
print(answer)
```

---

## 📋 Step 4: Structured Event Logging (`logs/events.json`)

All requests and responses entering and leaving the LLM server, agent microservices, and Web App are automatically audited:
- **Location**: `logs/events.json`
- **Audit Coverage**: Full HTTP methods, paths, timestamps, execution durations, status codes, and complete request/response JSON payloads.
- **Size Cap & Purging**: The log file is strictly capped at **100MB**. When a new event would cause the file to exceed 100MB, older events are purged in **FIFO** order using file-locked concurrency control.
- **Live Viewing**: Tab 3 of the Web App provides a real-time table of recent events with JSON previews.

---

## 📊 Agent Options Comparison

| Feature | Direct Mode (`:8000`) | Option 1: `agents/agent_server.py` (`:8001`) | Option 2: `agents/langchain_agent.py` |
|---|---|---|---|
| **Role** | Raw LLM Inference Engine | Front-end Agent Microservice | Python Agent Pipeline |
| **Tools / Skills** | None (Raw Completion) | `calculate`, `lookup_system_status` | `get_current_time`, `calculate` |
| **Interface** | OpenAI REST API (`/v1`) | REST API (`/agent/chat`, `/health`) | Python Function / CLI |
| **Protocol** | HTTP on `:8000` | HTTP on `:8001` | In-process Python invocation |
| **Best For** | Private RAG, embeddings, direct completions | Web frontends, mobile apps, microservices | Python workflows, notebooks, pipelines |

---

## 🛑 Step 5: Shut Down

### 1. Stopping the Agent Server (`agents/agent_server.py`)
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

### 3. Stopping the Web App Console (`web_app.py`)
- If running in foreground: Press `Ctrl + C`
- If running in background:
  - **Linux / macOS**:
    ```bash
    kill $(lsof -t -i:8002)
    ```
  - **Windows (PowerShell)**:
    ```powershell
    Stop-Process -Id (Get-NetTCPConnection -LocalPort 8002).OwningProcess -Force
    ```

---

## 🛠 Technical Notes & Troubleshooting

- **Single Active Agent Rule**:
  In the Web Console and agent orchestration layer, only 1 agent is permitted to run at a time to prevent port contention, resource fragmentation, and GPU VRAM exhaustion.
- **Llama 3.2 Single-Tool-Call Template Rule**:
  Meta's official chat template for `Llama-3.2-3B-Instruct` enforces that each assistant turn contains at most one tool call (`"This model only supports single tool-calls at once!"`). Both `agents/agent_server.py` and `agents/langchain_agent.py` are structured to execute tool calls sequentially to ensure strict compatibility.
- **Offline / Air-Gapped Operation**:
  The flags `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` ensure no connection attempts are made to the public internet.
- **Structured Audit Logging**:
  All incoming queries and responses are recorded to `logs/events.json` with thread/process-safe file locking and an automatic 100MB max FIFO purge.
