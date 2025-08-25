# app.py — Sahayak (Indic Study Buddy) — Mistral-only
# Summary / Flashcards / Quiz; replies in same language; robust Mistral fallback

import traceback
import streamlit as st
from huggingface_hub import InferenceClient


# ---------------------------- PAGE CONFIG ----------------------------
st.set_page_config(page_title="Sahayak — Indic Study Buddy (Mistral)", page_icon="📚")

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

# ---------------------------- TOKEN ----------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN")
if not HF_TOKEN:
    st.error('Missing HF token. In Streamlit: Settings → Secrets → add\n\nHF_TOKEN = "hf_xxx"\n\nthen Restart.')
    st.stop()

# ---------------------------- MISTRAL-ONLY CANDIDATES ----------------------------
# We will try these Mistral-family models in order until one works on the serverless API.
MISTRAL_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    
]

def get_mistral_client():
    if "ACTIVE_MODEL" in st.session_state and "HF_CLIENT" in st.session_state:
        return st.session_state.HF_CLIENT, st.session_state.ACTIVE_MODEL

    last_err = None
    for mid in MISTRAL_MODELS:
        try:
            cli = InferenceClient(model=mid, token=HF_TOKEN)
            # Tiny probe to confirm the endpoint exists
            _ = cli.text_generation("ping", max_new_tokens=1, return_full_text=False, stream=False)
            st.session_state.ACTIVE_MODEL = mid
            st.session_state.HF_CLIENT = cli
            return cli, mid
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"No Mistral serverless endpoint responded. Last error: {type(last_err).__name__}: {last_err}")

try:
    client, ACTIVE_MODEL = get_mistral_client()
except Exception as e:
    st.error("Failed to initialize any Mistral endpoint.")
    st.code(traceback.format_exc())
    st.stop()

with st.sidebar:
    st.markdown("### Settings")
    st.write("Using Mistral model:", f"`{ACTIVE_MODEL}`")
    st.caption("The app auto-selects the first available Mistral endpoint.")

# ---------------------------- HELPERS ----------------------------
def build_prompt(system_prompt: str, history: list[tuple[str, str]], user_block: str) -> str:
    lines = []
    if system_prompt.strip():
        lines.append("System:\n" + system_prompt.strip())
    if history:
        for u, a in history[-3:]:  # keep last 3 turns
            lines.append("User:\n" + (u or ""))
            lines.append("Assistant:\n" + (a or ""))
    lines.append("User:\n" + user_block.strip())
    lines.append("Assistant:\n")
    return "\n\n".join(lines)

def call_llm_text_generation(prompt: str) -> str:
    return client.text_generation(
        prompt,
        max_new_tokens=700,
        temperature=0.4,
        return_full_text=False,
        stream=False,
    )

# ---------------------------- UI ----------------------------
st.title("📚 Sahayak — Indic Study Buddy (Mistral)")
st.caption("Paste your study text, choose a mode, and get a Summary, Flashcards, or Quiz — in the same language you use.")

mode = st.radio("Choose a mode:", ["Summary", "Flashcards", "Quiz"], horizontal=True)

if "history" not in st.session_state:
    st.session_state.history = []  # list of (user_text, assistant_text)

with st.form("study_form", clear_on_submit=False):
    user_text = st.text_area("Paste your study material here:", height=220, placeholder="Type in Hindi, English, or Hinglish…")
    submitted = st.form_submit_button(f"Generate {mode}")

# ---------------------------- RUN ----------------------------
if submitted and user_text.strip():
    user_block = f"Mode: {mode}\n\nText:\n{user_text}\n\nReturn the result strictly in the format for {mode}."
    full_prompt = build_prompt(SYSTEM_PROMPT, st.session_state.history, user_block)

    try:
        with st.spinner(f"Generating with {ACTIVE_MODEL}…"):
            reply = call_llm_text_generation(full_prompt)
        st.session_state.history.append((f"[{mode}] {user_text.strip()}", reply))
    except Exception as e:
        st.error(f"Model call failed: {type(e).__name__}: {e}")
        st.code(traceback.format_exc())

# ---------------------------- OUTPUT ----------------------------
if st.session_state.history:
    latest_user, latest_reply = st.session_state.history[-1]
    st.subheader(f"{mode} Output")
    st.write(latest_reply)

st.markdown("---")
st.markdown("**Tip:** Paste at least 150–200 words. Sahayak will respond in the same language as your input.")

st.markdown("Developed by [Devam]")
