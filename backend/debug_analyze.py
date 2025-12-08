"""Debug attention analysis."""

import sys
sys.path.insert(0, '/Users/ankitsanjyal/Desktop/Diffusion-Detective/backend')

from app.pipeline import AttentionStore
import torch

# Create a simple test
store = AttentionStore()

# Add a simple 3D attention map
attention_map = torch.rand(16, 4096, 77)  # [heads, spatial, text_tokens]
store.add_attention_map(5, attention_map)

# Test tokenization
tokens = ['<|startoftext|>', 'a', 'red', 'apple', '<|endoftext|>'] + ['<|pad|>'] * 72

print(f"Attention maps stored: {list(store.attention_maps.keys())}")
print(f"Tokens: {len(tokens)}")

# Try to analyze
result = store.analyze_attention(5, tokens, "Test Phase", False)

print(f"\nAnalysis result: {result}")

if result:
    print(f"  Token: {result['token']}")
    print(f"  Score: {result['score']}")
    print(f"  Message: {result['message']}")
else:
    print("  NO RESULT!")
    
    # Debug the analyze function
    attention_list = store.attention_maps[5]
    print(f"\n  Debug: attention_list has {len(attention_list)} maps")
    for i, attn in enumerate(attention_list):
        print(f"    Map {i}: dim={attn.dim()}, shape={attn.shape}")
