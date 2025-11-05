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
DEVICE ="cpu"

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
model_base_f32 = AutoModelForCausalLM.from_pretrained(MODEL_BASE, torch_dtype=torch.float32, low_cpu_mem_usage=True)
model_f32 = PeftModel.from_pretrained(model_base_f32, LORA_PATH).merge_and_unload()

base_model_int8 = AutoModelForCausalLM.from_pretrained(MODEL_BASE, torch_dtype=torch.float32, low_cpu_mem_usage=True)
model_int8 = PeftModel.from_pretrained(base_model_int8, LORA_PATH).merge_and_unload()
model_int8 = torch.quantization.quantize_dynamic(
    model_int8,
    {torch.nn.Linear},
    dtype=torch.qint8
)

model_int8 = model_int8.to(DEVICE)
model_int8.eval()

# ----------------------
# UTILITAIRES
# ----------------------
def format_prompt(text: str) -> str:
    text = text[:600].strip()
    return f"Article: {text}\n\nGenerate a clear, complete summary of this article in 12–15 words:"

def extract_summary(full_text: str) -> str:
    if "Generate a clear, complete summary of this article in 12–15 words:" in full_text:
        full_text = full_text.split("Generate a clear, complete summary of this article in 12–15 words:")[-1]
        
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
    model = model_int8 if optimized else model_f32
        
    prompt = format_prompt(text)

    tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False)
    tracker.start()
    start_time = time.time()

    try:
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

        
        with torch.no_grad():
            if optimized:
                summary_ids = model.generate(
                **inputs,
                max_new_tokens=20,           # ⚡ Moins de tokens = plus rapide
                min_new_tokens=10,           
                do_sample=False,             # ⚡ Greedy = plus rapide
                num_beams=1,                 # ⚡ Pas de beam search
                repetition_penalty=1.15,     # Léger pour éviter répétitions
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                )
            else:
                # Baseline : privilégier la qualité
                summary_ids = model.generate(
                **inputs,
                max_new_tokens=30,
                min_new_tokens=10,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                )

        full_output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Supprimer le prompt si le modèle de base l’a reproduit
        if not optimized and full_output.startswith(prompt):
            full_output = full_output[len(prompt):].strip()

        summary = extract_summary(full_output)



        if count_words(summary) > 15:
            summary = " ".join(summary.split()[:15])

    except Exception as e:
        tracker.stop()
        return {"error": f"Erreur pendant la génération : {str(e)}"}

    elapsed_time = time.time() - start_time
    try:
        tracker.stop()
        e = tracker.final_emissions_data
        # energy_consumed est déjà en Wh selon le code de référence
        energy_wh = float(getattr(e, "energy_consumed", 0.0))
        # Optionnel : récupérer aussi le CO2
    except Exception as ex:
        print(f"Erreur lors de la récupération de l'énergie: {ex}")
        energy_wh = 0.0
    
    return {
        "summary": summary,
        "latency_ms": round(elapsed_time * 1000, 1),
        "energy_wh": round(energy_wh, 6)
    }
