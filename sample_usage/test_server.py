import os
import sys
import requests

port = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].isdigit() else os.environ.get("PORT", "8000")
BASE_URL = os.environ.get("OPENAI_BASE_URL", f"http://127.0.0.1:{port}/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "your-internal-secure-gateway-token-xyz")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def test_models():
    print(f"Checking GET {BASE_URL}/models ...")
    try:
        resp = requests.get(f"{BASE_URL}/models", headers=headers, timeout=5)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.json()}")
        return resp.json().get("data", [{}])[0].get("id", None)
    except Exception as e:
        print(f"Connection error: {e}")
        return None


def test_chat(model_id: str):
    print(f"\nSending test prompt to POST {BASE_URL}/chat/completions ...")
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Respond with: 'Private LLM server is up and running!'"}
        ],
        "max_tokens": 50,
        "temperature": 0.0
    }
    try:
        resp = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60)
        print(f"Status Code: {resp.status_code}")
        res_json = resp.json()
        print(f"Response: {res_json}")
        if "choices" in res_json:
            print("\nGenerated Message:")
            print(res_json["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"Request error: {e}")


if __name__ == "__main__":
    model = test_models()
    if model:
        test_chat(model)
    else:
        print("Could not retrieve model list. Is the server running?")
        sys.exit(1)
