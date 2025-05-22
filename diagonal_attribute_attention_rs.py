import torch
import torch.nn as nn
import torch.nn.functional as F

class DiagonalAttributeAttention(nn.Module):
    """
    Probes the persona attention heads by using an attribute query.
    """
    def __init__(self, embed_dim, output_dim, num_slots=2, drop_prob=0.1, ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.drop_prob = drop_prob
        self.scale = embed_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, output_dim)
        self.q_projs = nn.ModuleList([nn.Linear(embed_dim, output_dim) for _ in range(num_slots)])        
        
        self.v_projs = nn.ModuleList([  nn.Sequential(nn.Linear(embed_dim, embed_dim*2),
                                        nn.GELU(),
                                        nn.Linear(embed_dim*2, output_dim)) for _ in range(num_slots)])
    
    def forward(self, instance, attr, temperature=1):
        """
        Args:
            instance: Tensor of shape [B, n, D] (persona representations).
            attr_query: Tensor of shape [B, n, D] (the attribute query, repeated over tokens).
        Returns:
            decoded_attr: Tensor of shape [B, D] from the probe.
            attn_weights: Tensor of shape [B, n] with attention weights.
        """
        query = torch.stack([proj(attr) for proj in self.q_projs], dim=1)
        key = self.k_proj(instance)
        key = key.unsqueeze(1).repeat(1, self.num_slots, 1, 1)

        value_k = torch.stack([proj(instance) for proj in self.v_projs], dim=1)
        value_q = torch.stack([proj(attr) for proj in self.v_projs], dim=1)

        sim = (key * query).sum(dim=-1) * self.scale  # [B, n]
        attn_weights = F.softmax(sim/temperature, dim=-1)  # [B, s, n]
        attended = torch.einsum('bsn,bsnd->bsd', attn_weights, value_k)
        attended_q = torch.einsum('bsn,bsnd->bsd', attn_weights, value_q)

        return attended, attended_q, attn_weights
