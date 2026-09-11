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
                                 │                │   (agent_server.py on :8001) │
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

## 🛠️ Step 0: Download Models
1. Go to huggingface.co to find the model you would like to use. Some models requires you to make a request access it. The example here is for meta-llama/Meta-Llama-3.2-3B-Instruct.
    - After the model access is granted, go back to huggingface.co again and 
2. Create a token by following the steps below:
    - Click on the user icon (top-right) -> Access Tokens
    - Click on +Create new token (top-right)
    - Copy the token. It will not be shown again.

3. Go to the terminal in the IDE or regular shell terminal.
    ```
    mkdir ~/Documents/model_serv
    cd ~/Documents/model_serv
    ```
4. Install Python virtual environment and activate it.
    ```
    sudo apt install python3-venv
    python3 -m venv .venv
    source .venv/bin/activate
    ```
5. Install important libraries
    ```
    pip install -U transformers torch==2.13.0 "huggingface_hub[cli]"
    ```
6. Check Nvidia processor
    ```
    nvidia-smi
    ```
7. Authenticate with Huggingface using API key
    ```
    hf auth login --token <paste the token here>
    ```
8. Make the folder to store the models and download the model:
    ```
    mkdir models
    hf download meta-llama/Llama-3.2-3B-Instruct --local-dir ./models/Llama-3.2-3B-Instruct
    ```


## 🚀 Step 1: Start the Local Model Server

Before running direct queries or starting an agent, launch the core inference server.

### Virtual Environment (Recommended)

Default (port 8000):
```bash
cd /home/<uname>/Documents/model_serv
./.venv/bin/python serve_private_llm.py
```

Custom port (using `-l <port>`):
```bash
./.venv/bin/python serve_private_llm.py -l 8500
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
    "model": "Llama-3.2-3B-Instruct",
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

### Option 1: FastAPI Microservice Front (`agent_server.py`)

A persistent HTTP service listening on port `8001`. It receives user prompts, passes them to the local model alongside registered tool schemas, intercepts function calls, executes the local Python functions, and returns the synthesized answer.

#### Available Tools
- `calculate(expression)`: Mathematical calculations.
- `lookup_system_status()`: Inspects hardware and system metrics.

#### 1. Start the Agent Server
In a separate terminal:

**Default ports (listen: 8001, model: 8000):**
```bash
./.venv/bin/python agent_server.py
```

**Custom ports (specify listening port `-l` and/or model server port `-m`):**
```bash
# Listen on 8001, connect to model server on custom port 8500
./.venv/bin/python agent_server.py -m 8500

# Listen on custom port 9001, connect to model server on default port 8000
./.venv/bin/python agent_server.py -l 9001

# Specify both listening and model ports
./.venv/bin/python agent_server.py -l 9001 -m 8500
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

### Option 2: LangChain Python Agent (`langchain_agent.py`)

A Python-native agent using `langchain_core` and `langchain_openai`. Ideal for integrating directly into larger Python backends, background workflows, or automation pipelines.

#### Available Tools
- `get_current_time()`: Retrieves formatted system timestamp.
- `calculate(expression)`: Computes arithmetic expressions.

#### 1. Run the Agent CLI Test
```bash
./.venv/bin/python langchain_agent.py
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
from langchain_agent import run_agent

answer = run_agent("What time is it right now and what is 250 * 4?")
print(answer)
```

---

## 📊 Agent Options Comparison

| Feature | Direct Mode (`:8000`) | Option 1: `agent_server.py` (`:8001`) | Option 2: `langchain_agent.py` |
|---|---|---|---|
| **Role** | Raw LLM Inference Engine | Front-end Agent Microservice | Python Agent Pipeline |
| **Tools / Skills** | None (Raw Completion) | `calculate`, `lookup_system_status` | `get_current_time`, `calculate` |
| **Interface** | OpenAI REST API (`/v1`) | REST API (`/agent/chat`, `/health`) | Python Function / CLI |
| **Protocol** | HTTP on `:8000` | HTTP on `:8001` | In-process Python invocation |
| **Best For** | Private RAG, embeddings, direct completions | Web frontends, mobile apps, microservices | Python workflows, notebooks, pipelines |

---

## 🛑 Step 4: Shut Down

### 1. Stopping the Agent Server (`agent_server.py`)
- If running in foreground: Press `Ctrl + C`
- If running in background:
  ```bash
  kill $(lsof -t -i:8001)
  ```

### 2. Stopping the Model Server (`serve_private_llm.py`)
- If running in foreground: Press `Ctrl + C`
- If running in background:
  ```bash
  pkill -f "serve_private_llm.py"
  # Or: kill $(lsof -t -i:8000)
  ```
- If running via Docker Compose:
  ```bash
  docker compose down
  ```

---

## 🛠 Technical Notes & Troubleshooting

- **Llama 3.2 Single-Tool-Call Template Rule**:
  Meta's official chat template for `Llama-3.2-3B-Instruct` enforces that each assistant turn contains at most one tool call (`"This model only supports single tool-calls at once!"`). Both `agent_server.py` and `langchain_agent.py` are structured to execute tool calls sequentially to ensure strict compatibility.
- **Offline / Air-Gapped Operation**:
  The flags `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` ensure no connection attempts are made to the public internet.
- **Privacy & Prompt Logging**:
  `--no-enable-log-requests` guarantees that sensitive prompts, retrieved documents, and tool payloads remain in volatile memory and are never written to disk logs.
