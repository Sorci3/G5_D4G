import os
import torch
import time
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from codecarbon import EmissionsTracker

# ----------------------
# CONFIG
# ----------------------
MODEL_BASE = "EleutherAI/pythia-70m-deduped"
LORA_PATH = "./py/pythia-70m-xsum-summarizer"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ----------------------
# TOKENIZER
# ----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
tokenizer.pad_token = tokenizer.eos_token

# ----------------------
# CHARGEMENT DES MODÈLES
# ----------------------
model_base = AutoModelForCausalLM.from_pretrained(MODEL_BASE, torch_dtype=torch.float32, low_cpu_mem_usage=True)

base_for_lora = AutoModelForCausalLM.from_pretrained(MODEL_BASE, torch_dtype=torch.float32, low_cpu_mem_usage=True)
model_lora = PeftModel.from_pretrained(base_for_lora, LORA_PATH).merge_and_unload()

# ----------------------
# UTILITAIRES
# ----------------------
def format_prompt(text: str) -> str:
    text = text[:600].strip()
    return f"Article: {text}\n\nSummary (12 words):"

def extract_summary(full_text: str) -> str:
    if "Summary (12 words):" in full_text:
        full_text = full_text.split("Summary (12 words):")[-1]
    return full_text.strip().split("\n")[0]

def count_words(text: str) -> int:
    return len(text.split())

# ----------------------
# FONCTION PRINCIPALE
# ----------------------
def summarize_text(text: str, optimized: bool = False) -> dict:
    if not text.strip():
        return {"error": "Aucun texte fourni."}

    text = text[:4000]
    model = model_lora if optimized else model_base
    prompt = format_prompt(text) if optimized else f"Summarize in 12 words: {text[:600]}"

    tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False)
    tracker.start()
    start_time = time.time()

    try:
        model = model.to(DEVICE)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                min_new_tokens=8,
                num_beams=2,
                temperature=0.7,
                do_sample=False,
                early_stopping=True,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        summary = extract_summary(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

        if count_words(summary) > 15:
            summary = " ".join(summary.split()[:15])

    except Exception as e:
        tracker.stop()
        return {"error": f"Erreur pendant la génération : {str(e)}"}

    elapsed_time = time.time() - start_time
    try:
        emissions = tracker.stop() or 0.0
    except:
        emissions = 0.0

    return {
        "summary": summary,
        "latency_ms": round(elapsed_time * 1000, 1),
        "energy_wh": round(float(emissions), 6)
    }
