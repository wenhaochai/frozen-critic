"""Character-level 0.5B actor (owner, 2026-10-09): dense qwen3_5, random init, its own 100-token vocabulary.

Vocabulary (ids): 0 <|endoftext|> (stop), 1 <|im_start|>, 2 <|im_end|>, 3 newline, 4-98 the 95 printable ASCII
characters, 99 "boxed". Text is split into "boxed" and single characters, so the actor reads the ASCII-normalized
GSM8K prompt one character at a time, and writes \\boxed{ in three tokens. Every actor token is exactly one token of
the critic's vocabulary: critic_ids.json maps actor id -> critic id, so the frozen critic scores every actor token.
The actor may write only the 27 ids of allowed_ids.json (digits, + - * / = . , ( ) $, space, newline, \\, boxed, {, },
stop), masked in SGLang and in the trainer (miles/utils/fc_vocab_mask.py).
Shape: models/proxy-q35dense-0.5b (hidden 1024, SwiGLU 3072, 8 heads x 256, GDN 16/16 heads) with 24 layers
(18 Gated DeltaNet + 6 gated attention), untied embeddings, vocab padded to 128. Writes
  models/actor-q35d05b-char-text  Qwen3_5ForCausalLM            -> Megatron (--megatron-hf-checkpoint, --ref-load)
  models/actor-q35d05b-char-lm    Qwen3_5ForConditionalGeneration -> SGLang (--hf-checkpoint)
Run on della-vis1: .venv-critic/bin/python src/make_char_actor.py
"""
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models/proxy-q35dense-0.5b"
CRITIC_TOK = ROOT / "models/actor-q35d1b-lm"  # the critic's tokenizer files (vocab 151,667)
TEXT, LM = ROOT / "models/actor-q35d05b-char-text", ROOT / "models/actor-q35d05b-char-lm"
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
TOKENS = SPECIAL + ["\n"] + [chr(c) for c in range(0x20, 0x7F)] + ["boxed"]
OUT_CHARS = list("0123456789+-*/=.,()$ \n\\{}") + ["boxed", "<|endoftext|>"]
N_LAYERS, VOCAB_PAD = 24, 128
# the critic's chat template without tools: same prompt string, so the critic reads exactly what it read before
CHAT = ("{{- '<|endoftext|>' }}{%- if messages[0]['role'] == 'system' %}{{- '<|im_start|>system\\n' + messages[0]['content'] "
        "+ '<|im_end|>\\n' }}{%- else %}{{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}{%- endif %}"
        "{%- for message in messages %}{%- if not (message.role == 'system' and loop.first) %}{{- '<|im_start|>' + "
        "message.role + '\\n' + message.content + '<|im_end|>\\n' }}{%- endif %}{%- endfor %}"
        "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}")


def build_tokenizer():
    tok = Tokenizer(models.WordLevel({t: i for i, t in enumerate(TOKENS)}, unk_token="<|endoftext|>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"boxed|[\s\S]"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    tok.add_special_tokens(SPECIAL)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", pad_token="<|endoftext|>",
                                   unk_token="<|endoftext|>", model_max_length=8192, chat_template=CHAT)
    return fast


if __name__ == "__main__":
    fast = build_tokenizer()
    critic = AutoTokenizer.from_pretrained(CRITIC_TOK)
    cmap = []
    for t in TOKENS:
        e = [critic.convert_tokens_to_ids(t)] if t in SPECIAL else critic.encode(t, add_special_tokens=False)
        assert len(e) == 1, (t, e)
        cmap.append(e[0])
    assert len(set(cmap)) == len(cmap)
    # round trips: prompt text, the boxed answer, the chat template against the critic's
    q = [{"role": "user", "content": "Natalia sold 48/2 = 24 clips.\n\nPlease reason step by step, and put your final answer within \\boxed{}."}]
    s = fast.apply_chat_template(q, tokenize=False, add_generation_prompt=True)
    assert s == critic.apply_chat_template(q, tokenize=False, add_generation_prompt=True), s
    ids = fast.encode(s, add_special_tokens=False)
    assert fast.decode(ids) == s and 99 in ids and ids[0] == 0, ids[:5]
    assert critic.decode([cmap[i] for i in ids]) == s
    allowed = sorted(fast.convert_tokens_to_ids(c) for c in OUT_CHARS)
    assert len(allowed) == 27 and len(set(allowed)) == 27 and 0 in allowed
    a = fast.encode("48/2=24\n\\boxed{72}", add_special_tokens=False)
    assert set(a) <= set(allowed) and fast.decode(a) == "48/2=24\n\\boxed{72}"

    torch.manual_seed(0)
    tcfg = json.loads((SRC / "config.json").read_text())
    tcfg.update(num_hidden_layers=N_LAYERS, vocab_size=VOCAB_PAD, tie_word_embeddings=False, eos_token_id=0, pad_token_id=0,
                bos_token_id=None, layer_types=["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(N_LAYERS)])
    tcfg["rope_parameters"] = json.loads((ROOT / "configs/qwen3.5-0.8b.config.json").read_text())["text_config"]["rope_parameters"]
    cfg = AutoConfig.for_model(**{k: v for k, v in tcfg.items() if k != "model_type"}, model_type="qwen3_5_text")
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).to(torch.bfloat16)  # fp32 init: see make_actor_ckpt.py
    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    assert all(torch.isfinite(v.float()).all() for v in sd.values())
    n = sum(v.numel() for v in sd.values())
    emb = sd["model.embed_tokens.weight"].numel() + sd["lm_head.weight"].numel()
    print(f"{n/1e9:.3f}B params ({(n - emb)/1e9:.3f}B without embeddings), {len(sd)} tensors")
    full = json.loads((ROOT / "configs/qwen3.5-0.8b.config.json").read_text())
    for d in (TEXT, LM):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        fast.save_pretrained(d)
        (d / "generation_config.json").write_text(json.dumps({"eos_token_id": 0, "pad_token_id": 0}))
        (d / "critic_ids.json").write_text(json.dumps(cmap))
        (d / "allowed_ids.json").write_text(json.dumps(allowed))
    (TEXT / "config.json").write_text(json.dumps(dict(tcfg, architectures=["Qwen3_5ForCausalLM"], model_type="qwen3_5_text"), indent=1))
    save_file(sd, TEXT / "model.safetensors", metadata={"format": "pt"})
    json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in sd}}, open(TEXT / "model.safetensors.index.json", "w"))
    lcfg = dict(full, text_config=dict(tcfg, model_type="qwen3_5_text"), language_model_only=True, tie_word_embeddings=False,
                architectures=["Qwen3_5ForConditionalGeneration"])
    (LM / "config.json").write_text(json.dumps(lcfg, indent=1))
    lsd = {(k.replace("model.", "model.language_model.", 1) if k.startswith("model.") else k): v for k, v in sd.items()}
    save_file(lsd, LM / "model.safetensors", metadata={"format": "pt"})
    json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in lsd}}, open(LM / "model.safetensors.index.json", "w"))
    print("wrote", TEXT, LM, "allowed", [TOKENS[i] for i in allowed])
