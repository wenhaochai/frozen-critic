"""Random-init Phase 1 actor: dense qwen3_5 (the Qwen3.8-27B architecture) at about 1B, critic vocabulary.

Owner, 2026-10-07: start Phase 1 with this architecture (dense Qwen3.8-Flash-Next has no working miles/SGLang path yet).
Shape = models/proxy-q35dense-1b (hidden 1280, 24 layers: 18 Gated DeltaNet + 6 gated attention, SwiGLU 3840,
10 heads x 256, GDN 16 key / 16 value heads), embeddings untied (the path miles has run for Qwen3.5-9B), no MTP.
Writes two directories with the same weights (the labs-molt text-only recipe, tools/make_qwen35_text_ckpts.py):
  models/actor-q35d1b-text  Qwen3_5ForCausalLM, names model.*            -> Megatron via the bridge (--megatron-hf-checkpoint)
  models/actor-q35d1b-lm    Qwen3_5ForConditionalGeneration, language_model_only, names model.language_model.*
                            -> SGLang (--hf-checkpoint); it registers only the ConditionalGeneration classes.
Both carry the critic's tokenizer files and chat template, and generation stops at <|endoftext|> (151643).
"""
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models/proxy-q35dense-1b"
TEXT, LM = ROOT / "models/actor-q35d1b-text", ROOT / "models/actor-q35d1b-lm"
TOK = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
       "special_tokens_map.json", "chat_template.jinja", "generation_config.json"]

torch.manual_seed(0)
tcfg = json.loads((SRC / "config.json").read_text())
tcfg["tie_word_embeddings"] = False
# Keep Qwen3.5's own rope_parameters (interleaved mrope sections): with text only all three position ids are equal, so it
# equals 1D RoPE, and it is the form SGLang's ConditionalGeneration class and the labs-molt Qwen3.5 runs use.
tcfg["rope_parameters"] = json.loads((ROOT / "configs/qwen3.5-0.8b.config.json").read_text())["text_config"]["rope_parameters"]
cfg = AutoConfig.for_model(**{k: v for k, v in tcfg.items() if k != "model_type"}, model_type="qwen3_5_text")
# Initialise in float32, then cast: in bf16, A_log = log(uniform(0, 16)) draws exact zeros and gives -inf (seen in
# smoke job 15167571: NaN gradients at step 0 from 3 Gated DeltaNet layers).
model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).to(torch.bfloat16)
sd = {k: v.contiguous() for k, v in model.state_dict().items()}
bad = [k for k, v in sd.items() if not torch.isfinite(v.float()).all()]
assert not bad, f"non-finite tensors after init: {bad}"
n = sum(v.numel() for v in sd.values())
print(f"{type(model).__name__}: {n/1e9:.3f}B params, {len(sd)} tensors; lm_head tied: {model.config.tie_word_embeddings}")
assert "lm_head.weight" in sd and sd["lm_head.weight"].data_ptr() != sd["model.embed_tokens.weight"].data_ptr()

for d in (TEXT, LM):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in TOK:
        shutil.copy2(SRC / f, d / f)

# Megatron side: the text model as transformers saves it.
(TEXT / "config.json").write_text(json.dumps(dict(tcfg, architectures=["Qwen3_5ForCausalLM"], model_type="qwen3_5_text"), indent=1))
save_file(sd, TEXT / "model.safetensors", metadata={"format": "pt"})
json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in sd}}, open(TEXT / "model.safetensors.index.json", "w"))

# SGLang side: the ConditionalGeneration wrapper with the vision tower switched off.
full = json.loads((ROOT / "configs/qwen3.5-0.8b.config.json").read_text())
lcfg = dict(full, text_config=dict(tcfg, model_type="qwen3_5_text"), language_model_only=True, tie_word_embeddings=False,
            architectures=["Qwen3_5ForConditionalGeneration"])
(LM / "config.json").write_text(json.dumps(lcfg, indent=1))
lsd = {(k.replace("model.", "model.language_model.", 1) if k.startswith("model.") else k): v for k, v in sd.items()}
save_file(lsd, LM / "model.safetensors", metadata={"format": "pt"})
json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in lsd}}, open(LM / "model.safetensors.index.json", "w"))
print("wrote", TEXT, LM, "example keys:", sorted(lsd)[:3])
