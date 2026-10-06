"""Actor configs: the Qwen3.8-Flash-Next architecture (qwen4_exp) made dense, with the critic's vocabulary.

Owner, 2026-10-05: use the newest architecture miles supports, Qwen3.8-Flash-Next, but dense and about 1B.
From configs/qwen3.8-flash-next.config.json (HF sha de4b8e4d) we keep the block design: 3 Gated DeltaNet layers
then 1 gated full-attention layer with the QSA sparse indexer, 4-stream hyper-connections, one per-layer n-gram
embedding (PLE) at layer 2, partial rotary 0.25, sigmoid output gate. We change: experts removed (num_experts 0, so
vLLM builds a dense SwiGLU MLP of width intermediate_size), widths and depth scaled down, the n-gram table cut from
20M to 65,536 rows per head, MTP removed, mrope removed (with text only it equals 1D RoPE), embeddings tied, and the
vocabulary set to the critic's 151,667 ids (Qwen2.5 + <think> 151665, </think> 151666) with the critic's tokenizer.

Also writes a qwen3_5 dense proxy of each size (Qwen3_5ForCausalLM, same width, depth, MLP and heads, no
hyper-connections, indexer or PLE): transformers 5.6.2 has no qwen4_exp, so actor-update speed is measured on it.
Weights are not written: vLLM's load_format="dummy" gives random weights; Phase 1 makes a real random init.
"""
import copy
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRITIC = ROOT / "models" / "limite-1b-value-model"
VOCAB, EOS = 151667, 151643
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
                   "special_tokens_map.json", "chat_template.jinja"]

# name: hidden, layers, ffn, attention heads (head_dim 256), GDN value heads (dim 128)
# GDN value heads must be a multiple of the 16 key heads (vLLM kernel check).
SIZES = {"0.5b": (1024, 16, 3072, 8, 16), "1b": (1280, 24, 3840, 10, 16),
         "2b": (2048, 24, 6144, 16, 32), "4b": (2560, 32, 7680, 20, 48)}


def layer_types(n):
    return ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n)]


def q4_dense(d, L, ffn, heads, vh):
    src = json.loads((ROOT / "configs/qwen3.8-flash-next.config.json").read_text())["text_config"]
    c = copy.deepcopy(src)
    for k in ("num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size", "router_aux_loss_coef",
              "output_router_logits", "mtp", "mtp_use_dedicated_embeddings"):
        c.pop(k, None)
    c.update({
        "architectures": ["Qwen4ExpForCausalLM"], "model_type": "qwen4_exp_text",
        "hidden_size": d, "num_hidden_layers": L, "layer_types": layer_types(L), "intermediate_size": ffn,
        "num_experts": 0, "num_attention_heads": heads, "num_key_value_heads": 2, "head_dim": 256,
        "linear_num_key_heads": 16, "linear_key_head_dim": 128, "linear_num_value_heads": vh, "linear_value_head_dim": 128,
        "hc_lowrank": d // 8, "ple_embed_dim": d, "ple_layer_ids": [2], "ngram_vocab_size_base": 65536,
        "mtp_num_hidden_layers": 0, "max_position_embeddings": 8192, "tie_word_embeddings": True,
        "vocab_size": VOCAB, "eos_token_id": EOS, "bos_token_id": EOS, "pad_token_id": EOS,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25},
        "torch_dtype": "bfloat16",
    })
    return c


def q35_proxy(d, L, ffn, heads, vh):
    src = json.loads((ROOT / "configs/qwen3.5-0.8b.config.json").read_text())["text_config"]
    c = copy.deepcopy(src)
    c.pop("mtp_num_hidden_layers", None)
    c.update({
        "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text",
        "hidden_size": d, "num_hidden_layers": L, "layer_types": layer_types(L), "intermediate_size": ffn,
        "num_attention_heads": heads, "num_key_value_heads": 2, "head_dim": 256,
        "linear_num_key_heads": 16, "linear_key_head_dim": 128, "linear_num_value_heads": vh, "linear_value_head_dim": 128,
        "max_position_embeddings": 8192, "tie_word_embeddings": True, "vocab_size": VOCAB,
        "eos_token_id": EOS, "bos_token_id": EOS, "pad_token_id": EOS, "torch_dtype": "bfloat16",
    })
    rp = c.get("rope_parameters") or {}
    c["rope_parameters"] = {"rope_type": "default", "rope_theta": rp.get("rope_theta", 10000000),
                            "partial_rotary_factor": rp.get("partial_rotary_factor", 0.25)}
    return c


def write(dirname, cfg):
    out = ROOT / "models" / dirname
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "generation_config.json").write_text(json.dumps({"eos_token_id": EOS, "pad_token_id": EOS}))
    for f in TOKENIZER_FILES:
        shutil.copy2(CRITIC / f, out / f)
    return out


if __name__ == "__main__":
    for name, dims in SIZES.items():
        print(write(f"actor-q4dense-{name}", q4_dense(*dims)))
        print(write(f"proxy-q35dense-{name}", q35_proxy(*dims)))
