# app.py — Sahayak (Indic Study Buddy) — simple UI, Cloud-safe
# Uses HF text_generation (robust) + same-language replies + Summary/Flashcards/Quiz modes

import os
import traceback
import streamlit as st
from huggingface_hub import InferenceClient

# ---------------------------- PAGE CONFIG ----------------------------
st.set_page_config(page_title="Sahayak — Indic Study Buddy", page_icon="📚")

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

# ---------------------------- TOKEN & MODEL --------------------------
def resolve_hf_token():
    # 1) Streamlit secrets (Cloud or local .streamlit/secrets.toml)
    tok = st.secrets.get("HF_TOKEN", None)
    if tok:
        return tok
    # 2) Environment variables (if you prefer)
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    # 3) Optional manual paste in sidebar (set below)
    return st.session_state.get("HF_TOKEN", None)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # reliable text-generation endpoint

with st.sidebar:
    st.header("Settings")
    model_id = st.text_input("Model ID", value=DEFAULT_MODEL, help="Try Qwen2.5-7B, Zephyr-7B, Falcon-7B, etc.")
    pasted = st.text_input("HF Token (paste here if not in Secrets)", type="password")
    if pasted:
        st.session_state["HF_TOKEN"] = pasted
    HF_TOKEN = resolve_hf_token()
    st.write("HF token found:", bool(HF_TOKEN))
    st.caption("Tip: Add token in Settings → Secrets as: HF_TOKEN = \"hf_xxx\"")

# Create client (no chat API assumptions)
client = InferenceClient(model=model_id, token=HF_TOKEN)

# ---------------------------- HELPERS --------------------------
def build_prompt(system_prompt: str, history: list[tuple[str, str]], user_block: str) -> str:
    """
    Convert (system + history + user) into a single text prompt for text_generation.
    history: list of (user_text, assistant_text)
    """
    lines = []
    if system_prompt.strip():
        lines.append("System:\n" + system_prompt.strip())
    if history:
        for u, a in history[-3:]:  # keep last 3 turns to stay concise
            lines.append("User:\n" + (u or ""))
            lines.append("Assistant:\n" + (a or ""))
    lines.append("User:\n" + user_block.strip())
    lines.append("Assistant:\n")
    return "\n\n".join(lines)

def call_llm_text_generation(prompt: str) -> str:
    """
    Use serverless-friendly text_generation. No streaming here (simpler for Cloud).
    """
    return client.text_generation(
        prompt,
        max_new_tokens=700,
        temperature=0.4,
        return_full_text=False,
        stream=False,
    )

# ---------------------------- UI --------------------------
st.title("📚 Sahayak — Indic Study Buddy")
st.caption("Paste your study text, choose a mode, and get a Summary, Flashcards, or Quiz — in the same language you use.")

mode = st.radio("Choose a mode:", ["Summary", "Flashcards", "Quiz"], horizontal=True)

if "history" not in st.session_state:
    st.session_state.history = []  # list of (user_text, assistant_text)

with st.form("study_form", clear_on_submit=False):
    user_text = st.text_area("Paste your study material here:", height=220, placeholder="Type in Hindi, English, or Hinglish…")
    submitted = st.form_submit_button(f"Generate {mode}")

# ---------------------------- RUN --------------------------
if submitted and user_text.strip():
    if not HF_TOKEN:
        st.error("Missing Hugging Face token. Add it in Streamlit: Settings → Secrets → `HF_TOKEN = \"hf_xxx\"`")
    else:
        user_block = f"Mode: {mode}\n\nText:\n{user_text}\n\nReturn the result strictly in the format for {mode}."
        full_prompt = build_prompt(SYSTEM_PROMPT, st.session_state.history, user_block)

        try:
            with st.spinner("Generating…"):
                reply = call_llm_text_generation(full_prompt)
            st.session_state.history.append((f"[{mode}] {user_text.strip()}", reply))
        except Exception as e:
            st.error(f"Model call failed: {type(e).__name__}: {e}")
            st.code(traceback.format_exc())

# ---------------------------- OUTPUT --------------------------
if st.session_state.history:
    latest_user, latest_reply = st.session_state.history[-1]
    st.subheader(f"{mode} Output")
    st.write(latest_reply)

st.markdown("---")
st.markdown("**Tip:** For best results, paste at least 150–200 words. Sahayak will respond in the same language as your input.")

