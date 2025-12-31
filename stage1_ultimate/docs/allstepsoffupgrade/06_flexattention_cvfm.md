## FlexAttention for Trained CVFM (Optional)

**Goal**: 2-3× faster attention when num_views >5 or learned spatial weighting needed.

**When**: Only if num_views >5 or you add learned attention between views.

**Implementation**: `src/streetvision/tta/flex_cvfm.py`
```python
from torch.nn.attention.flex_attention import flex_attention

class CVFMFlexAttention(nn.Module):
    def __init__(self):
        self.flex_attn = torch.compile(
            flex_attention,
            mode='max-autotune',
            dynamic=True
        )
    
    def forward(self, view_features):
        # Custom score_mod for spatial proximity
        def score_mod(score, b, h, q_idx, kv_idx):
            return score * spatial_weight[q_idx, kv_idx]
        
        return self.flex_attn(
            query=view_features,
            key=view_features,
            value=view_features,
            score_mod=score_mod,
            is_gqa=True
        )
```

**Recommendation**: Skip for now (2-5 views). Use simple weighted mean. Add only if view count grows or you need learned fusion.

