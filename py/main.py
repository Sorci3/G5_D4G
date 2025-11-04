import os
import random
import numpy as np
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from codecarbon import EmissionsTracker

os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

MODEL_NAME = "EleutherAI/pythia-70m-deduped"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def summarize_text(text: str, optimized: bool = False) -> dict:
    if not text.strip():
        return {"error": "Aucun texte fourni."}

    tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False)
    tracker.start()
    start_time = time.time()

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        local_model = model.to(device)

        if optimized and device == "cpu":
            local_model = torch.quantization.quantize_dynamic(
                local_model, {torch.nn.Linear}, dtype=torch.qint8
            )

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(device)

        summary_ids = local_model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=4,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    except Exception as e:
        tracker.stop()
        return {"error": f"Erreur pendant le résumé : {str(e)}"}

    elapsed_time = time.time() - start_time
    try:
        emissions = tracker.stop() or 0.0
    except:
        emissions = 0.0

    energy_wh = round(float(emissions), 6)
    latency_ms = round(elapsed_time * 1000, 1)

    return {
        "summary": summary,
        "energy_wh": energy_wh,
        "latency_ms": latency_ms
    }