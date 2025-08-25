# app.py — Sahayak (Indic Study Buddy) — single model: Mistral-7B-Instruct-v0.3
# Modes: Summary / Flashcards / Quiz
# Same-language replies; robust HF token resolution; chat->text fallback; optional debug

import os
import traceback
import streamlit as st
from huggingface_hub import InferenceClient
from huggingface_hub.utils._errors import HfHubHTTPError

# ---------------------------- PAGE CONFIG ----------------------------
st.set_page_config(page_title="Sahayak — Indic Study Buddy (Mistral v0.3)", page_icon="📚", layout="centered")

# ---------------------------- SYSTEM PROMPT --------------------------
SYSTEM_PROMPT = """
You are “Sahayak”, an open-source Indic Study Buddy.
Always reply in the same language as the input (Hindi, English, or Hinglish),
unless the user clearly asks for a different language.

You support three modes:
1) Summary: Summarize the provided text in 4–6 clear bullet points.
2) Flashcards: Produce 4–6 Question–Answer pairs in this format:
   Q1: ...
   A1: ...
   Q2: ...
   A2: ...
3) Quiz: Produce 4–6 MCQs in this format:
   Q1: ...
   A) ...
   B) ...
   C) ...
   D) ...
   Correct: B

Guidelines:
- Be concise, clear, and beginner-friendly.
- If the input is unclear or too short, ask a brief clarifying question first.
- Be respectful and neutral. Do not give definitive medical, legal, or financial advice.
"""

# ---------------------------- TOKEN & MODEL ----------------------------
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # fixed, single model

def resolve_hf_token() -> str | None:
    # 1) Streamlit Secrets (Cloud or local .streamlit/secrets.toml)
    tok = st.secrets.get("HF_TOKEN") if hasattr(st, "secrets") else None
    if tok:
        return tok
    # 2) Environment (useful for local dev)
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    return None

with st.sidebar:
    st.header("Settings")
    show_debug = st.toggle("Show debug errors", value=False, help="Print full Python tracebacks on errors")
    HF_TOKEN = resolve_hf_token()
    st.write("HF token found:", bool(HF_TOKEN))
    st.caption('If False, set Secrets → `HF_TOKEN = "hf_xxx"` (Cloud) or create `.streamlit/secrets.toml` locally.')

if not HF_TOKEN:
    st.error(
        "Missing Hugging Face token.\n\n"
        "• Streamlit Cloud: Settings → Secrets → add `HF_TOKEN = \"hf_xxx\"`, then Restart.\n"
        "• Local: create `.streamlit/secrets.toml` with the same line."
    )
    st.stop()

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# ---------------------------- HELPERS ----------------------------
def call_llm(system_prompt: str, history: list[tuple[str, str]], user_block: str, max_tokens: int = 700, temperature: float = 0.4) -> str:
    """
    1) Try chat_completion (if supported by the endpoint).
    2) Fallback to text_generation with a simple chat template.
    """
    # Build chat messages
    messages = [{"role": "system", "content": system_prompt}]
    for u, a in history[-3:]:  # keep last 3 turns
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_block})

    # Try chat_completion first
    try:
        out = client.chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature)
        return out.choices[0].message["content"]
    except Exception as e1:
        if show_debug:
            st.error(f"chat_completion failed: {type(e1).__name__}: {e1}")
            st.code(traceback.format_exc())
        # Fallback to text_generation with a flattened prompt
        lines = []
        for m in messages:
            if m["role"] == "system":
                lines.append("System:\n" + m["content"])
            elif m["role"] == "user":
                lines.append("User:\n" + m["content"])
            elif m["role"] == "assistant":
                lines.append("Assistant:\n" + m["content"])
        prompt = "\n\n".join(lines) + "\n\nAssistant:\n"
        try:
            return client.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False,
                stream=False,
            )
        except HfHubHTTPError as e2:
            raise e2
        except Exception as e3:
            raise e3

# ---------------------------- UI ----------------------------
st.title("📚 Sahayak — Indic Study Buddy (Mistral v0.3)")
st.caption("Paste your study text, choose a mode, and get a Summary, Flashcards, or Quiz — in the same language you use.")

mode = st.radio("Choose a mode:", ["Summary", "Flashcards", "Quiz"], horizontal=True)

if "history" not in st.session_state:
    st.session_state.history = []  # list of (user_text, assistant_text)

with st.form("study_form", clear_on_submit=False):
    user_text = st.text_area(
        "Paste your study material here:",
        height=240,
        placeholder="Type in Hindi, English, or Hinglish…",
    )
    submitted = st.form_submit_button(f"Generate {mode}")

# ---------------------------- RUN ----------------------------
if submitted and user_text.strip():
    user_block = (
        f"Mode: {mode}\n\n"
        f"Text:\n{user_text.strip()}\n\n"
        f"Return the result strictly in the format for {mode}."
    )
    try:
        with st.spinner(f"Generating with {MODEL_ID}…"):
            reply = call_llm(SYSTEM_PROMPT, st.session_state.history, user_block, max_tokens=700, temperature=0.4)
        st.session_state.history.append((f"[{mode}] {user_text.strip()}", reply))
    except Exception as e:
        st.error(f"Model call failed: {type(e).__name__}: {e}")
        if show_debug:
            st.code(traceback.format_exc())

# ---------------------------- OUTPUT ----------------------------
if st.session_state.history:
    latest_user, latest_reply = st.session_state.history[-1]
    st.subheader(f"{mode} Output")
    st.write(latest_reply)

st.markdown("---")
st.markdown("**Tip:** Paste at least 150–200 words. Sahayak will respond in the same language as your input.")

st.markdown("Developed by [Devam]")
