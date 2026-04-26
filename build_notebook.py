"""
Generates colab_inference.ipynb — Downloads inference results from GCS
and displays S-CoT vs Flat comparison.

No Tunix/TPU code here. Inference runs on the TPU via run_inference.py
and results are synced to GCS as JSON files.
"""
import json

def md_cell(lines):
    """Create a markdown cell from a list of lines."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" if not line.endswith("\n") else line for line in lines[:-1]] + [lines[-1]]
    }

def code_cell(code_text):
    """Create a code cell from a multi-line string."""
    lines = code_text.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

notebook = {
    "cells": [
        md_cell([
            "# \U0001f9e0 S-CoT Distilled Model \u2014 Results & Analysis",
            "",
            "This notebook downloads inference results generated **on the TPU** and displays them.",
            "",
            "> **Why not run inference here?** The training used [Tunix](https://github.com/google/tunix),",
            "> a **TPU-only** JAX framework. The LoRA checkpoints are in Orbax format.",
            "> Inference runs automatically on the TPU after training and results sync to GCS.",
            "",
            "**Two models trained on TPU v6e-8:**",
            "| Model | Final Loss | Perplexity | Dataset |",
            "|-------|-----------|------------|---------|",
            "| `sft-scot` (S-CoT reasoning) | 1.27 | 3.55 | 3,817 |",
            "| `sft-flat` (Flat baseline) | 0.275 | 1.32 | 3,681 |",
        ]),

        code_cell("""\
# Cell 1: Authenticate & download results from GCS
from google.colab import auth
auth.authenticate_user()

import os, subprocess

!mkdir -p /content/scot_results

# Download everything from the GCS bucket
print("Downloading from gs://tpu-builder1-scot-checkpoints/ ...")
!gsutil -m cp -r gs://tpu-builder1-scot-checkpoints/* /content/scot_results/ 2>/dev/null || true

# Show what we got
total = 0
for root, dirs, files in os.walk('/content/scot_results'):
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path)
        total += size
        print(f'  {path} ({size:,} bytes)')

if total == 0:
    print("\\n\u26a0\ufe0f  Bucket is empty! Inference hasn't run yet.")
    print("Run the watchdog and wait for training + inference to complete.")
else:
    print(f"\\n\u2705 Downloaded {total:,} bytes total")\
"""),

        code_cell("""\
# Cell 2: Display helper + S-CoT results
import json, os
from IPython.display import HTML, display

CARD_STYLE = '''
<div style="border:1px solid #555; border-radius:10px; padding:18px; margin:14px 0;
            background:linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
    <div style="color:#e94560; font-weight:bold; font-size:15px;
                border-bottom:1px solid #333; padding-bottom:8px; margin-bottom:10px;">
        \U0001f4ac Q: {question}
    </div>
    <div style="color:#eee; margin-top:8px; white-space:pre-wrap;
                font-family:'Fira Code',monospace; font-size:13px;
                line-height:1.6; max-height:400px; overflow-y:auto;">
{answer}
    </div>
    <div style="color:#888; margin-top:10px; font-size:11px;
                border-top:1px solid #333; padding-top:6px;">
        \U0001f4ca {num_tokens} tokens &bull; {time_seconds}s &bull; {model}
    </div>
</div>
'''

def find_results(name):
    \"\"\"Search common paths for inference results.\"\"\"
    candidates = [
        f'/content/scot_results/inference_{name}.json',
        f'/content/scot_results/tpu-builder1-scot-checkpoints/inference_{name}.json',
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None

def show_results(data, title, emoji='\U0001f52c'):
    if not data:
        print(f'{title}: No results found yet.')
        return
    display(HTML(f'<h2 style="color:#e94560;">{emoji} {title}</h2>'))
    for r in data:
        display(HTML(CARD_STYLE.format(**r)))

# --- Show S-CoT Results ---
scot_data = find_results('scot')
show_results(scot_data, 'S-CoT Structured Reasoning', '\U0001f9e0')\
"""),

        code_cell("""\
# Cell 3: Display Flat baseline results
flat_data = find_results('flat')
show_results(flat_data, 'Flat Baseline', '\U0001f4d6')\
"""),

        code_cell("""\
# Cell 4: Side-by-side comparison
scot = find_results('scot')
flat = find_results('flat')

if scot and flat:
    display(HTML('<h2 style="color:#e94560;">\u2694\ufe0f Side-by-Side: S-CoT vs Flat</h2>'))
    for s, f_item in zip(scot, flat):
        display(HTML(f'''
        <div style="border:1px solid #555; border-radius:10px; padding:18px; margin:14px 0;
                    background:linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
            <div style="color:#e94560; font-weight:bold; font-size:15px;
                        border-bottom:1px solid #444; padding-bottom:8px; margin-bottom:12px;">
                \U0001f4ac Q: {s["question"]}
            </div>
            <div style="display:flex; gap:16px; margin-top:8px;">
                <div style="flex:1; background:rgba(0,210,255,0.08); padding:14px;
                            border-radius:8px; border-left:3px solid #00d2ff;">
                    <div style="color:#00d2ff; font-weight:bold; margin-bottom:8px;">
                        \U0001f9e0 S-CoT ({s["num_tokens"]} tok, {s["time_seconds"]}s)
                    </div>
                    <div style="color:#eee; white-space:pre-wrap; font-family:monospace;
                                font-size:12px; line-height:1.5; max-height:300px;
                                overflow-y:auto;">{s["answer"][:600]}</div>
                </div>
                <div style="flex:1; background:rgba(255,165,0,0.08); padding:14px;
                            border-radius:8px; border-left:3px solid #ffa500;">
                    <div style="color:#ffa500; font-weight:bold; margin-bottom:8px;">
                        \U0001f4d6 Flat ({f_item["num_tokens"]} tok, {f_item["time_seconds"]}s)
                    </div>
                    <div style="color:#eee; white-space:pre-wrap; font-family:monospace;
                                font-size:12px; line-height:1.5; max-height:300px;
                                overflow-y:auto;">{f_item["answer"][:600]}</div>
                </div>
            </div>
        </div>
        '''))
else:
    print('Need both S-CoT and Flat results for comparison.')
    print('Run the watchdog to completion and re-download.')\
"""),

        code_cell("""\
# Cell 5: Training metrics dashboard
from IPython.display import HTML, display

display(HTML('''
<h2 style="color:#e94560;">\U0001f4ca Training Summary</h2>
<table style="border-collapse:collapse; margin:12px 0; font-family:'Fira Code',monospace;
              width:100%; max-width:700px;">
  <tr style="background:linear-gradient(90deg, #16213e, #0f3460);">
    <th style="padding:10px 16px; border:1px solid #444; color:#e94560; text-align:left;">Metric</th>
    <th style="padding:10px 16px; border:1px solid #444; color:#00d2ff; text-align:center;">\U0001f9e0 S-CoT</th>
    <th style="padding:10px 16px; border:1px solid #444; color:#ffa500; text-align:center;">\U0001f4d6 Flat</th>
  </tr>
  <tr><td style="padding:8px 16px; border:1px solid #333; color:#ccc;">Base Model</td>
      <td style="padding:8px 16px; border:1px solid #333; color:#eee; text-align:center;">Qwen2.5-3B-Instruct</td>
      <td style="padding:8px 16px; border:1px solid #333; color:#eee; text-align:center;">Qwen2.5-3B-Instruct</td></tr>
  <tr style="background:#1a1a2e;">
      <td style="padding:8px 16px; border:1px solid #333; color:#ccc;">LoRA Config</td>
      <td style="padding:8px 16px; border:1px solid #333; color:#eee; text-align:center;">rank=16, \u03b1=32</td>
      <td style="padding:8px 16px; border:1px solid #333; color:#eee; text-align:center;">rank=16, \u03b1=32</td></tr>
  <tr><td style="padding:8px 16px; border:1px solid #333; color:#ccc;">Training Steps</td>
      <td style="padding:8px 16px; border:1px solid #333; color:#eee; text-align:center;">500</td>
      <td style="padding:8px 16px; border:1px solid #333; color:#eee; text-align:center;">500</td></tr>
  <tr style="background:#1a1a2e;">
      <td style="padding:8px 16px; border:1px solid #333; color:#ccc;">Dataset Size</td>
      <td style="padding:8px 16px; border:1px solid #333; color:#eee; text-align:center;">3,817</td>
      <td style="padding:8px 16px; border:1px solid #333; color:#eee; text-align:center;">3,681</td></tr>
  <tr style="background:linear-gradient(90deg, #1a1a2e, #16213e);">
      <td style="padding:10px 16px; border:1px solid #444; color:#e94560; font-weight:bold;">\u2b07 Final Loss</td>
      <td style="padding:10px 16px; border:1px solid #444; color:#00d2ff; font-weight:bold; text-align:center; font-size:16px;">1.27</td>
      <td style="padding:10px 16px; border:1px solid #444; color:#ffa500; font-weight:bold; text-align:center; font-size:16px;">0.275</td></tr>
  <tr style="background:linear-gradient(90deg, #16213e, #1a1a2e);">
      <td style="padding:10px 16px; border:1px solid #444; color:#e94560; font-weight:bold;">Perplexity</td>
      <td style="padding:10px 16px; border:1px solid #444; color:#00d2ff; font-weight:bold; text-align:center; font-size:16px;">3.55</td>
      <td style="padding:10px 16px; border:1px solid #444; color:#ffa500; font-weight:bold; text-align:center; font-size:16px;">1.32</td></tr>
  <tr><td style="padding:8px 16px; border:1px solid #333; color:#ccc;">Hardware</td>
      <td style="padding:8px 16px; border:1px solid #333; color:#eee; text-align:center;">TPU v6e-8</td>
      <td style="padding:8px 16px; border:1px solid #333; color:#eee; text-align:center;">TPU v6e-8</td></tr>
  <tr style="background:#1a1a2e;">
      <td style="padding:8px 16px; border:1px solid #333; color:#ccc;">Training Time</td>
      <td style="padding:8px 16px; border:1px solid #333; color:#eee; text-align:center;">~2m 51s</td>
      <td style="padding:8px 16px; border:1px solid #333; color:#eee; text-align:center;">~2m 51s</td></tr>
  <tr><td style="padding:8px 16px; border:1px solid #333; color:#ccc;">Speed</td>
      <td style="padding:8px 16px; border:1px solid #333; color:#eee; text-align:center;">~4.2 steps/s</td>
      <td style="padding:8px 16px; border:1px solid #333; color:#eee; text-align:center;">~4.2 steps/s</td></tr>
</table>
'''))\
"""),
    ],
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open("colab_inference.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("✅ Notebook colab_inference.ipynb generated successfully!")
