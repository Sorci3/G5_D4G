"""
Fine-tuning de EleutherAI/pythia-70m-deduped sur XSum pour des résumés de 10-15 mots.
Reprise automatique possible depuis une LoRA existante.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
OUTPUT_DIR = "./pythia-70m-xsum-summarizer"
MAX_LENGTH = 256

# =====================================================
# CHARGEMENT DU TOKENIZER
# =====================================================
print("🔹 Chargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# =====================================================
# CHARGEMENT DU MODÈLE (NOUVEAU OU REPRISE)
# =====================================================
print("🔹 Chargement du modèle...")

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
    print(" Reprise de l'entraînement depuis la LoRA existante...")
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

    #  Assure que les couches LoRA sont bien entraînables
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

else:
    print(" Nouveau fine-tuning à partir du modèle de base...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)

# Vérification du nombre de paramètres entraînables
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"🔧 Paramètres entraînables : {trainable:,} / {total:,} ({trainable/total:.2%})")
model.print_trainable_parameters()

# =====================================================
# CHARGEMENT DU DATASET XSUM
# =====================================================
print("🔹 Chargement du dataset XSum...")
dataset = load_dataset("xsum", split="train")
eval_dataset = load_dataset("xsum", split="validation")

# =====================================================
# PRÉTRAITEMENT DES DONNÉES
# =====================================================
def preprocess_function(examples):
    """
    Utilise le MÊME format de prompt qu'en inférence
    """
    inputs = []
    for document, summary in zip(examples["document"], examples["summary"]):
        #  PROMPT IDENTIQUE à celui du main.py
        # Limite à 600 caractères comme en inférence
        text = document[:600].strip()
        
        prompt = f"Article: {text}\n\nGenerate a clear, complete summary of this article in 12–15 words: {summary}"
        inputs.append(prompt)
    
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    return model_inputs

print("🔹 Préparation du dataset...")
tokenized_train = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)
tokenized_eval = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names
)

# =====================================================
# DATA COLLATOR
# =====================================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# =====================================================
# ARGUMENTS D’ENTRAÎNEMENT
# =====================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,  # augmente à 3+ si tu veux pousser davantage
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,  # plus petit si tu reprends l'entraînement
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=50,
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    report_to="none",
    remove_unused_columns=False,
)

# =====================================================
# TRAINER
# =====================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# =====================================================
# ENTRAÎNEMENT
# =====================================================
print(" Lancement de l'entraînement...")
trainer.train()

# =====================================================
# SAUVEGARDE DU MODÈLE
# =====================================================
print(" Sauvegarde du modèle...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f" Modèle sauvegardé dans {OUTPUT_DIR}")

# =====================================================
# TEST DE GÉNÉRATION
# =====================================================
def generate_summary(text, max_new_tokens=30):
    """Test avec le MÊME prompt qu'en inférence"""
    text = text[:600].strip()
    prompt = f"Article: {text}\n\nGenerate a clear, complete summary of this article in 12–15 words:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=15,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraire le résumé
    if full_text.startswith(prompt):
        full_text = full_text[len(prompt):].strip()
    
    if "Generate a clear, complete summary of this article in 12–15 words:" in full_text:
        full_text = full_text.split("Generate a clear, complete summary of this article in 12–15 words:")[-1]
    
    summary = full_text.strip().split("\n")[0]
    return summary


print("\n Test sur un exemple :")
sample = eval_dataset[0]["document"]
print(f"Texte original (extrait): {sample[:200]}...")
print(f"\nRésumé généré: {generate_summary(sample)}")
print(f"Résumé de référence: {eval_dataset[0]['summary']}")
