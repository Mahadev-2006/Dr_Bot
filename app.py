

Models required
---------------
  - meta-llama/Llama-3.2-3B-Instruct  (HuggingFace, gated – HF token needed)
  - valhalla/distilbart-mnli-12-3      (public)
  - cross-encoder/ms-marco-MiniLM-L-6-v2 (public)
  - FremyCompany/BioLORD-2023          (public)
  - swin_base_patch4_window7_224       (timm, pretrained)
  - models/swin_aptos_best.pth         (fine-tuned weights – train with train_swin.py)
"""

import os
import gc
import warnings

import torch
import gradio as gr
import numpy as np
from PIL import Image
from timm import create_model
from torchvision import transforms
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------------------------------------------------------
# 1. Global configuration
# ---------------------------------------------------------------------------
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN        = os.getenv("HF_TOKEN", "")           # set via env variable
CLASSIFIER_PATH = "models/swin_aptos_best.pth"
CLASS_NAMES     = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

# ---------------------------------------------------------------------------
# 2. Load models
# ---------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading models…")

# Swin Transformer classifier
classifier_model = create_model(
    "swin_base_patch4_window7_224", pretrained=False, num_classes=5
)
classifier_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
classifier_model.to(DEVICE).eval()

classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# LLaMA 3.2 3B (4-bit quantised)
_llama_id        = "meta-llama/Llama-3.2-3B-Instruct"
llama_tokenizer  = AutoTokenizer.from_pretrained(_llama_id, token=HF_TOKEN)
llama_model      = AutoModelForCausalLM.from_pretrained(
    _llama_id,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
)
llama_pipe = pipeline("text-generation", model=llama_model, tokenizer=llama_tokenizer)

# DistilBART-MNLI semantic intent router
intent_router = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-3",
    device=0 if DEVICE == "cuda" else -1,
)

print("✅ All models loaded.")

# ---------------------------------------------------------------------------
# 3. DR classification
# ---------------------------------------------------------------------------
def predict_dr(image: Image.Image) -> str:
    """Run Swin Transformer inference and return predicted DR stage name."""
    img_tensor = classifier_transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = classifier_model(img_tensor)
        idx    = output.argmax(1).item()
    return CLASS_NAMES[idx]


# ---------------------------------------------------------------------------
# 4. RAG re-ranking
# ---------------------------------------------------------------------------
def _rerank_docs(query: str, candidates: list, top_k: int = 3) -> list:
    """Re-rank hybrid RAG candidates with the CrossEncoder (from retriever)."""
    from src.retriever import reranker
    pairs  = [[query, c[0]] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = np.argsort(scores)[::-1]
    return [candidates[i] for i in ranked[:top_k]]


# ---------------------------------------------------------------------------
# 5. LLM response generation
# ---------------------------------------------------------------------------
MEDICAL_SYSTEM_PROMPT = """\
You are DRBot, a compassionate medical assistant specialising in Diabetic Retinopathy and eye health.

STRICT RULES:
- Answer ONLY using the medical context provided below.
- If the context does not contain the answer, say exactly:
  "I don't have enough information on that — please consult your ophthalmologist."
- Never invent facts, drug names, or statistics not in the context.
- Use simple, empathetic language a non-medical person can understand.
- Always recommend seeing a doctor for next steps or treatment decisions.

Retrieved medical context:
{context}
"""


def get_medical_response(message: str, stage: str | None = None) -> str:
    """
    RAG → LLaMA pipeline.
    Works for ALL stages including No DR.
    """
    from src.retriever import hybrid_retrieve

    retrieval_query = f"{stage} diabetic retinopathy {message}" if stage else message
    candidates      = hybrid_retrieve(retrieval_query, top_k=15)
    top_docs        = _rerank_docs(message, candidates, top_k=3)
    context         = "\n\n---\n\n".join(doc[0] for doc in top_docs)

    system_content = MEDICAL_SYSTEM_PROMPT.format(context=context)
    user_content   = (
        f"My retinal scan result was: {stage}.\n\nMy question: {message}"
        if stage else message
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]
    prompt = llama_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    output = llama_pipe(prompt, max_new_tokens=300, temperature=0.3,
                        do_sample=True, repetition_penalty=1.1)
    return output[0]["generated_text"][len(prompt):].strip()


def get_general_response(message: str) -> str:
    messages = [
        {"role": "system", "content": (
            "You are DRBot, a friendly assistant. "
            "Do NOT provide medical diagnoses or advice. "
            "Keep responses brief and helpful."
        )},
        {"role": "user", "content": message},
    ]
    prompt = llama_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    output = llama_pipe(prompt, max_new_tokens=150, temperature=0.7, do_sample=True)
    return output[0]["generated_text"][len(prompt):].strip()


# ---------------------------------------------------------------------------
# 6. Gradio handler
# ---------------------------------------------------------------------------
def handle_submission(message: str, image, history: list, state: str):
    """
    Main event handler.

    Fixes applied vs. original prototype:
      FIX-1 : Image cleared after diagnosis so re-submissions don't repeat.
      FIX-2 : RAG runs for ALL stages, including No DR.
      FIX-3 : Only the initial No DR diagnosis is hardcoded; follow-ups use RAG.
    """
    if history is None:
        history = []
    current_stage = state

    # ── Image submitted ────────────────────────────────────────────────────
    if image is not None:
        pil_img       = Image.open(image)
        current_stage = predict_dr(pil_img)

        if current_stage == "No DR":
            reply = (
                "Great news — **No Diabetic Retinopathy** was detected in your scan.\n\n"
                "Keep managing your blood sugar and schedule your next annual screening. "
                "Feel free to ask me any questions about your eye health!"
            )
        else:
            reply = (
                f"I have analysed your retinal scan.\n\n"
                f"The detected stage is: **{current_stage}**.\n\n"
                "You can now ask me what this means, what to expect, "
                "or what your next steps should be."
            )

        history = history + [{"role": "assistant", "content": reply}]
        yield history, current_stage, "", None   # FIX-1: clear image widget
        return

    # ── Text message ──────────────────────────────────────────────────────
    if message and message.strip():
        history = history + [{"role": "user", "content": message}]
        yield history, current_stage, "", None

        result     = intent_router(message, ["medical question", "general greeting"])
        top_intent = result["labels"][0]

        # FIX-2: RAG always runs (even for No DR patients)
        if top_intent == "medical question":
            answer = get_medical_response(message, stage=current_stage)
        else:
            answer = get_general_response(message)

        history = history + [{"role": "assistant", "content": answer}]
        yield history, current_stage, "", None


# ---------------------------------------------------------------------------
# 7. UI helpers
# ---------------------------------------------------------------------------
def update_vram() -> str:
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{used:.2f} GB / {total:.1f} GB"
    return "CPU mode"


def clear_vram() -> str:
    gc.collect()
    torch.cuda.empty_cache()
    return update_vram()


# ---------------------------------------------------------------------------
# 8. Gradio 5 UI
# ---------------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(), title="DRBot") as demo:

    gr.Markdown("## 👁️ DRBot — Multimodal Diabetic Retinopathy Assistant")
    gr.Markdown(
        "Upload a retinal fundus image for autonomous grading, "
        "or ask a question about Diabetic Retinopathy. "
        "Powered by **Swin Transformer** + **Hybrid RAG** (FAISS + BM25 + CrossEncoder) + **LLaMA 3.2**."
    )

    stage_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="DRBot conversation",
                bubble_full_width=False,
                height=520,
                type="messages",
                show_label=False,
            )

        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Retinal fundus image",
                type="filepath",
                sources=["upload"],
            )
            with gr.Accordion("System status", open=False):
                vram_status   = gr.Textbox(label="GPU memory", value=update_vram(), interactive=False)
                clear_mem_btn = gr.Button("Clear VRAM cache")

    with gr.Row():
        text_input = gr.Textbox(
            placeholder="Ask about your results, DR stages, symptoms, next steps…",
            label="Your message",
            scale=4,
        )
        submit_btn = gr.Button("Send", variant="primary", scale=1)

    # Event bindings
    submit_btn.click(
        fn=handle_submission,
        inputs=[text_input, image_input, chatbot, stage_state],
        outputs=[chatbot, stage_state, text_input, image_input],
    ).then(fn=update_vram, outputs=vram_status)

    text_input.submit(
        fn=handle_submission,
        inputs=[text_input, image_input, chatbot, stage_state],
        outputs=[chatbot, stage_state, text_input, image_input],
    ).then(fn=update_vram, outputs=vram_status)

    clear_mem_btn.click(fn=clear_vram, outputs=vram_status)


if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True, show_error=True, max_threads=2)
