"""
Fine-tuning de EleutherAI/pythia-70m-deduped sur XSum pour des résumés de 10-15 mots
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Configuration
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
OUTPUT_DIR = "./pythia-70m-xsum-summarizer"
MAX_LENGTH = 256



print(" Chargement du modèle et du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)


# Configuration LoRA pour un fine-tuning efficace
print(" Configuration de LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()



# Chargement du dataset XSum
print(" Chargement du dataset XSum...")
dataset = load_dataset("xsum", split="train",trust_remote_code=True)  # Limité pour l'exemple
eval_dataset = load_dataset("xsum", split="validation")


# Fonction de préparation des données
def preprocess_function(examples):
    inputs = []
    for document, summary in zip(examples["document"], examples["summary"]):
        prompt = f"""### Instruction:Résume le texte suivant en 10-15 word maximum.### Texte:{document[:500]}### Résumé:{summary}"""
        inputs.append(prompt)
    
    # Tokenization
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Les labels sont identiques aux inputs pour le language modeling
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    return model_inputs

# Préparation des datasets
print(" Préparation des données...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
)

tokenized_eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Arguments d'entraînement
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=50,
    eval_steps=500,
    save_steps=1000,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    report_to="none",
    remove_unused_columns=False,

)

# Création du Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
)

# Entraînement
trainer.train()

# Sauvegarde du modèle
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f" Modèle sauvegardé dans {OUTPUT_DIR}")

# Fonction de génération pour tester
def generate_summary(text, max_new_tokens=20):

    prompt = f"""### Instruction:Résume le texte suivant en 10-15 mots maximum.### Texte:{text[:500]}### Résumé:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = full_text.split("### Résumé:")[-1].strip()
    return summary

# Test sur un exemple
print("\n Test de génération:")
test_text = eval_dataset[0]["document"]
print(f"Texte original (extrait): {test_text[:200]}...")
print(f"\nRésumé généré: {generate_summary(test_text)}")
print(f"Résumé référence: {eval_dataset[0]['summary']}")