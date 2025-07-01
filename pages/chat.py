import os
import json
import requests
import streamlit as st
import union
from flytekit.configuration import Config

# ---------------- Sidebar -----------------
with st.sidebar:
    union_api_key = st.text_input("Union API Key", key="union_api_key", type="password")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max tokens", 16, 2048, 512)
    qwen_endpoint = st.text_input(
        "qwen-service endpoint",
        value=os.getenv("QWEN_ENDPOINT", "https://summer-glade-f277a.apps.serverless-1.us-east-2.s.union.ai"),
    )
    # --- Retrieval config --------------------------------------------------
    st.divider()
    st.markdown("**Retrieval settings (Paul-Graham essays)**")

    union_endpoint = st.text_input(
        "Union API endpoint",
        value=os.getenv("UNION_ENDPOINT", "https://serverless.union.ai"),
        help="https://<host> part of the URL in the Union UI.",
    )
    union_project = st.text_input("Project", value=os.getenv("UNION_PROJECT", "default"))
    union_domain = st.text_input("Domain", value=os.getenv("UNION_DOMAIN", "development"))

    retrieval_max_results = st.number_input(
        "Max retrieved chunks", min_value=1, max_value=5, value=3, step=1
    )
    retrieval_max_distance = st.slider(
        "Max distance", min_value=0.0, max_value=2.0, value=1.2, step=0.05
    )
    st.markdown("[Union docs](https://www.union.ai)")

# ---------------- Status indicator -----------------
# We intentionally *do not* use `st_autorefresh` here.  Auto-refreshing the
# page while a retrieval or LLM call is in flight would wipe the spinners and
# UI state.  Instead we run `service_ready()` once per script execution.

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

def _get_remote(endpoint: str, project: str, domain: str, api_key: str) -> union.UnionRemote:
    key = (endpoint, project, domain)
    if "_union_remote_cache" not in st.session_state:
        st.session_state["_union_remote_cache"] = {}
    cache = st.session_state["_union_remote_cache"]
    if key not in cache:
        cache[key] = union.UnionRemote.from_api_key(
            api_key.strip(),
            default_project=project,
            default_domain=domain,
            endpoint=endpoint,
        )
    return cache[key]


def _get_launch_plan(remote: union.UnionRemote):
    return remote.fetch_launch_plan(name="download-chunk-embed.main")


def query_essays_remote(
    query: str,
    max_results: int,
    endpoint: str,
    project: str,
    domain: str,
    api_key: str,
) -> list[dict]:
    """Execute the launch plan synchronously and return list-of-dict result."""

    remote = _get_remote(endpoint, project, domain, api_key)
    lp = _get_launch_plan(remote)
    exec = remote.execute(
        lp,
        inputs={"query": query, "max_results": max_results},
        wait=True,
        type_hints={"o0": str, "o1": list[dict]},
    )

    raw_docs = exec.outputs.get("o1") or []
    if raw_docs and isinstance(raw_docs[0], str):
        try:
            return [json.loads(x) for x in raw_docs]
        except Exception:
            return []
    return raw_docs


def remote_completion(history: list[dict[str, str]], context_docs: list[dict] | None = None) -> str | None:
    # Prepend context as a system message if available.
    messages = history.copy()
    if context_docs:
        context_text = "\n\n---\n\n".join(d.get("document", "") for d in context_docs)
        # Strict RAG instruction: answer ONLY from context.
        system_msg = {
            "role": "system",
            "content": (
                "You are a question-answering assistant. Use ONLY the information "
                "provided in the CONTEXT below to answer the user's question. If the answer "
                "is not contained in the context, reply with 'I don't know.'\n\n"
                "CONTEXT:\n" + context_text
            ),
        }
        messages = [system_msg] + messages

    payload = {
        "model": "qwen2",
        "messages": messages,
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
        # ----------------------------------------------------
        # 1) Retrieve relevant chunks for the prompt
        # ----------------------------------------------------
        docs: list[dict] = []
        if union_api_key and union_endpoint:
            try:
                with st.spinner("Retrieving context …"):
                    docs = query_essays_remote(
                        query=prompt,
                        max_results=int(retrieval_max_results),
                        endpoint=union_endpoint,
                        project=union_project,
                        domain=union_domain,
                        api_key=union_api_key,
                    )

                docs = [d for d in docs if d.get("distance", 1.0) <= retrieval_max_distance]
            except Exception as e:
                st.warning(f"RAG retrieval failed: {e}")

        # ----------------------------------------------------
        # 2) Send prompt + context to the LLM
        # ----------------------------------------------------
        reply = remote_completion(
            [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            context_docs=docs,
        ) or "❗️Error calling qwen-service."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        chat_container.chat_message("assistant").write(reply) 