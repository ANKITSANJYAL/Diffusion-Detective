"""Debug confidence calculation."""

import torch

# Simulate attention scores
attention_scores = torch.tensor([0.009, 0.008, 0.007, 0.006] + [0.001] * 73)  # 77 tokens

max_attention = attention_scores.max().item()
print(f"Max attention: {max_attention}")

# Get top 3
top_values, top_indices = torch.topk(attention_scores, 3)

print("\nTop 3 tokens:")
for idx, val in zip(top_indices, top_values):
    raw_score = val.item()
    relative_confidence = (raw_score / max_attention) * 100
    
    # Apply scaling
    if relative_confidence >= 98:
        confidence = 80 + (relative_confidence - 98) * 7.5
    elif relative_confidence >= 90:
        confidence = 65 + (relative_confidence - 90) * 1.9
    elif relative_confidence >= 75:
        confidence = 45 + (relative_confidence - 75) * 1.3
    elif relative_confidence >= 50:
        confidence = 20 + (relative_confidence - 50) * 1.0
    else:
        confidence = relative_confidence * 0.4
    
    confidence = min(max(confidence, 0), 95)
    
    print(f"  Token {idx}: raw={raw_score:.6f}, relative={relative_confidence:.2f}%, scaled={confidence:.1f}%")
