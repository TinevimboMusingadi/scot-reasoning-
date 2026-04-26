import json

path = r"c:\Users\Tinevimbo\scot-reasoning-\scot_colab_sft.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 6: LoRA target modules
cell_6 = nb["cells"][6]["source"]
for i, line in enumerate(cell_6):
    if '"q_proj", "k_proj", "v_proj", "o_proj",' in line:
        cell_6[i] = '    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",\n'
    if '"gate_proj", "up_proj", "down_proj"],' in line:
        cell_6[i] = '                      "gate_proj", "up_proj", "down_proj",\n'
        # Insertion after this line
        cell_6.insert(i + 1, '                      "embed_tokens", "lm_head"],\n')
        break

# 2. Update Cell 10: Training Args
cell_10 = nb["cells"][10]["source"]
for i, line in enumerate(cell_10):
    if 'max_steps = 100,' in line:
        cell_10[i] = line.replace('100', '750')
    if 'args = TrainingArguments(' in line:
        # Add num_train_epochs right after TrainingArguments(
        cell_10.insert(i+1, '        num_train_epochs = 3,\n')

# 3. Update Cell 12: Inference
cell_12 = nb["cells"][12]["source"]
# It already has tokenizer = tokenizer. I'll add eos_token_id for safety.
for i, line in enumerate(cell_12):
    if '_ = model.generate(' in line:
        cell_12[i] = line.replace('**inputs,', '**inputs, eos_token_id = tokenizer.eos_token_id,')

# 4. Update Cell 14: Save Model
cell_14 = nb["cells"][14]["source"]
for i, line in enumerate(cell_14):
    if 'drive.mount' in line or 'model.save_pretrained' in line or 'tokenizer.save_pretrained' in line:
        cell_14[i] = line.lstrip("# ").replace("#", "")

with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook Phase 1 improvements applied.")
