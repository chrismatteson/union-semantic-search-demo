import os
import json
import requests
import streamlit as st
import union
from streamlit_autorefresh import st_autorefresh  # type: ignore

# ---------------- Sidebar -----------------
with st.sidebar:
    union_api_key = st.text_input("Union API Key", key="union_api_key", type="password")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max tokens", 16, 2048, 512)
    qwen_endpoint = st.text_input(
        "qwen-service endpoint",
        value=os.getenv("QWEN_ENDPOINT", "https://summer-glade-f277a.apps.serverless-1.us-east-2.s.union.ai"),
    )
    st.markdown("[Union docs](https://www.union.ai)")

# ---------------- Status indicator -----------------
# Refresh every 5 seconds
st_autorefresh(interval=5000, key="qwen_status_tick_chat")

if "_qwen_status_box_chat" not in st.session_state:
    st.session_state["_qwen_status_box_chat"] = st.sidebar.empty()
status_box = st.session_state["_qwen_status_box_chat"]


def _render_status(msg: str, ok: bool):
    prev = st.session_state.get("_qwen_status_prev_chat")
    if msg == prev:
        return
    status_box.empty()
    (status_box.success if ok else status_box.error)(msg)
    st.session_state["_qwen_status_prev_chat"] = msg


def service_ready() -> bool:
    if not qwen_endpoint:
        _render_status("qwen-service: ❓ endpoint not set", False)
        return False
    try:
        r = requests.get(f"{qwen_endpoint}/v1/models", timeout=3)
        ok_json = r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json")
        if ok_json:
            _render_status("qwen-service: ✅ ready", True)
            return True
        _render_status(f"qwen-service: 🔴 {r.status_code}", False)
    except Exception:
        _render_status("qwen-service: 🔴 unreachable", False)
    return False


# Call once so status is populated on first load
service_ready()


# ---------------- Remote completion helper -----------------

def remote_completion(history: list[dict[str, str]]) -> str | None:
    if not qwen_endpoint:
        st.error("⚠️ Provide the qwen-service endpoint in the sidebar or QWEN_ENDPOINT env var.")
        return None

    payload = {
        "model": "qwen2",
        "messages": history,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    headers = {"Content-Type": "application/json"}
    if union_api_key:
        headers["Authorization"] = f"Bearer {union_api_key.strip()}"

    try:
        r = requests.post(
            f"{qwen_endpoint}/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.HTTPError as e:
        st.error(f"❗️HTTP {e.response.status_code}:\n```{e.response.text[:400]}```")
    except Exception as e:
        st.error(f"❗️Request failed: {type(e).__name__}: {e}")
    return None


# ---------------- Chat UI -----------------

st.header("Chat with Qwen")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I am Qwen. Ask me anything."}
    ]

chat_container = st.container()
for m in st.session_state.messages:
    chat_container.chat_message(m["role"]).write(m["content"])

prompt = st.chat_input(placeholder="Type your question here...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    chat_container.chat_message("user").write(prompt)

    if not service_ready():
        st.error("qwen-service is not available right now.")
    else:
        reply = remote_completion(
            [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        ) or "❗️Error calling qwen-service."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        chat_container.chat_message("assistant").write(reply) 