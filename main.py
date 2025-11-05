import os
import torch
import time
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
from peft import PeftModel
from codecarbon import EmissionsTracker

# CONFIG
MODEL_BASE = "EleutherAI/pythia-70m-deduped"
LORA_PATH = "./py/pythia-70m-xsum-summarizer"
DEVICE = "cpu"

os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
tokenizer.pad_token = tokenizer.eos_token

# CHARGEMENT DES MODÈLES
model_base = AutoModelForCausalLM.from_pretrained(MODEL_BASE, torch_dtype=torch.float32, low_cpu_mem_usage=True)
base_for_lora = AutoModelForCausalLM.from_pretrained(MODEL_BASE, torch_dtype=torch.float32, low_cpu_mem_usage=True)
model_lora = PeftModel.from_pretrained(base_for_lora, LORA_PATH).merge_and_unload()

# UTILITAIRES
def format_prompt(text: str) -> str:
    text = text[:600].strip()
    return f"Article: {text}\n\nSummary (12 words):"

def extract_summary(full_text: str) -> str:
    if "Summary (12 words):" in full_text:
        full_text = full_text.split("Summary (12 words):")[-1]
    return full_text.strip().split("\n")[0]

def count_words(text: str) -> int:
    return len(text.split())

# FONCTION PRINCIPALE
def summarize_text(text: str, optimized: bool = False) -> dict:
    if not text.strip():
        return {"error": "Aucun texte fourni."}

    text = text[:4000]
    model = model_lora if optimized else model_base
    if optimized:
        quantize_(model, int8_dynamic_activation_int8_weight())
    prompt = format_prompt(text) if optimized else f"Summarize in 15 words: {text[:600]}"

    tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False)
    tracker.start()
    start_time = time.time()

    try:
        model = model.to(DEVICE)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

        with torch.no_grad():
            if optimized:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=20,          
                    min_new_tokens=15,          
                    num_beams=3,               
                    do_sample=True,
                    early_stopping=False,       
                    no_repeat_ngram_size=2,     
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=20,          
                    min_new_tokens=15,          
                    num_beams=3,               
                    do_sample=False,
                    early_stopping=False,       
                    no_repeat_ngram_size=2,     
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

            summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            summary = extract_summary(summary)

        if count_words(summary) > 15:
            summary = " ".join(summary.split()[:15])

    except Exception as e:
        tracker.stop()
        return {"error": f"Erreur pendant la génération : {str(e)}"}

    elapsed_time = time.time() - start_time
    try:
        emissions = tracker.stop() 
    except:
        emissions = 0.0

    return {
        "summary": summary,
        "latency_ms": round(elapsed_time * 1000, 1),
        "energy_wh": round(float(emissions), 6)
    }
