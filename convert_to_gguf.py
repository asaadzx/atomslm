#!/usr/bin/env python3
import os
import sys
import json
import shutil
import argparse
import subprocess
import torch
from safetensors.torch import save_file

def remap_key(old_key: str) -> str:
    # 1. Base mappings
    if old_key == "token_embedding.weight":
        return "model.embed_tokens.weight"
    if old_key == "final_norm.weight":
        return "model.norm.weight"
    if old_key == "lm_head.weight":
        return "lm_head.weight"
    
    # 2. Skip attention mask (llama.cpp creates this dynamically)
    if "attention.bias" in old_key:
        return None

    # 3. Layer specific mappings
    parts = old_key.split(".")
    if parts[0] == "layers" and len(parts) >= 3:
        idx = parts[1]
        rest = ".".join(parts[2:])

        # Norms
        if rest == "attention_norm.weight":
            return f"model.layers.{idx}.input_layernorm.weight"
        if rest == "ffn_norm.weight":
            return f"model.layers.{idx}.post_attention_layernorm.weight"

        # Attention
        if rest == "attention.q_proj.weight":
            return f"model.layers.{idx}.self_attn.q_proj.weight"
        if rest == "attention.k_proj.weight":
            return f"model.layers.{idx}.self_attn.k_proj.weight"
        if rest == "attention.v_proj.weight":
            return f"model.layers.{idx}.self_attn.v_proj.weight"
        if rest == "attention.out_proj.weight":
            return f"model.layers.{idx}.self_attn.o_proj.weight"

        # SwiGLU FFN
        if rest == "ffn.w1.weight":
            return f"model.layers.{idx}.mlp.gate_proj.weight"
        if rest == "ffn.w2.weight":
            return f"model.layers.{idx}.mlp.down_proj.weight"
        if rest == "ffn.w3.weight":
            return f"model.layers.{idx}.mlp.up_proj.weight"

    raise KeyError(f"Unrecognised key in state dict: {old_key}")

def main():
    parser = argparse.ArgumentParser(description="Unified PyTorch to Safetensors + GGUF Converter")
    parser.add_argument("--input", default="model/atom_slm.pt", help="Path to input PyTorch .pt model file")
    parser.add_argument("--hf_dir", default="model_hf", help="Output folder to store HuggingFace safetensors format")
    parser.add_argument("--gguf_dir", default="model_gguf", help="Output folder to store final GGUF file")
    parser.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0", "auto"], 
                        help="Quantization/precision format for the GGUF model")
    args = parser.parse_args()

    # 1. Setup Output Folders
    os.makedirs(args.hf_dir, exist_ok=True)
    os.makedirs(args.gguf_dir, exist_ok=True)

    print(f"[*] Loading PyTorch state dict from: {args.input}")
    if not os.path.exists(args.input):
        print(f"[!] Error: Input file '{args.input}' not found. Please train the model or specify the correct path.")
        sys.exit(1)

    state_dict = torch.load(args.input, map_location="cpu", weights_only=True)
    new_state_dict = {}

    print("[*] Remapping weights to HF Gemma format & adjusting RMSNorm scale offset...")
    for old_key, tensor in state_dict.items():
        new_key = remap_key(old_key)
        if new_key is None:
            print(f"  [Skipped] {old_key}")
            continue

        # Adjust RMSNorm scale weight offset: HF Gemma uses 1.0 + weight, so we subtract 1.0
        if new_key.endswith("norm.weight") or new_key.endswith("layernorm.weight"):
            tensor = tensor - 1.0
            print(f"  [Offset RMSNorm] {old_key} -> {new_key}")
        else:
            print(f"  {old_key} -> {new_key}")

        new_state_dict[new_key] = tensor

    # 2. Save Safetensors
    safetensors_path = os.path.join(args.hf_dir, "model.safetensors")
    print(f"[*] Saving safetensors file to: {safetensors_path}")
    save_file(new_state_dict, safetensors_path)

    # 3. Create config.json matching model parameters
    config = {
        "architectures": ["GemmaForCausalLM"],
        "model_type": "gemma",
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 682,
        "vocab_size": 256000,
        "rms_norm_eps": 1e-6,
        "head_dim": 32,
        "max_position_embeddings": 512,
        "torch_dtype": "float32"
    }
    config_path = os.path.join(args.hf_dir, "config.json")
    print(f"[*] Creating HuggingFace config.json: {config_path}")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # 4. Copy Tokenizer files
    tokenizer_source_dir = os.path.dirname(args.input)
    tokenizer_files = ["tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    print("[*] Copying tokenizer files to HF folder...")
    for filename in tokenizer_files:
        src = os.path.join(tokenizer_source_dir, filename)
        dst = os.path.join(args.hf_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {src} -> {dst}")
        else:
            if filename != "special_tokens_map.json":
                print(f"  [Warning] Tokenizer file '{src}' not found. Conversion might fail if missing.")

    # 5. Run llama.cpp conversion script
    outfile_path = os.path.join(args.gguf_dir, f"atom_slm_{args.outtype}.gguf")
    llama_cpp_converter = os.path.join("llm++", "llama.cpp", "convert_hf_to_gguf.py")
    
    if not os.path.exists(llama_cpp_converter):
        print(f"[!] Error: llama.cpp converter script not found at {llama_cpp_converter}")
        sys.exit(1)

    cmd = [
        sys.executable,
        llama_cpp_converter,
        args.hf_dir,
        "--outfile", outfile_path,
        "--outtype", args.outtype
    ]
    print(f"[*] Running llama.cpp GGUF converter: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"\n[+] SUCCESS! Final GGUF model saved to: {outfile_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Error running GGUF conversion script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
