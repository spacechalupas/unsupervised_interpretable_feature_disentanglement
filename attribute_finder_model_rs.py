import torch
import torch.nn as nn
from diagonal_attribute_attention_rs import DiagonalAttributeAttention 

class AttributeFinder(nn.Module):
    
    def __init__(self, head_dim, mid_dim,  num_features, num_heads, num_class, ds_factor, num_slots=2):#, num_sub_prototypes=5):
        super().__init__()

        self.head_encoders = nn.ModuleList([nn.Sequential(nn.Linear(head_dim, mid_dim*2, bias=False),
                                                        nn.GELU(),
                                                        nn.Linear(mid_dim*2, mid_dim, bias=False)) for _ in range(num_heads)])
        self.norm_inputs = nn.LayerNorm(mid_dim)
        self.head_grouper = DiagonalAttributeAttention(mid_dim, num_features, num_slots)
        self.shared_proj = nn.ModuleList([nn.Linear(num_features, num_features, bias=True) for _ in range(num_slots)])
        self.norm_slots = nn.LayerNorm(num_features)
        self.prototypes = nn.Linear(num_features, num_class, bias=True)


    def forward(self, heads_tensor, attr_tensor):
        """
        Args:
            heads_tensor: Tensor of shape (batch_size, num_heads, head_dim)
            attr_tensor: Tensor of shape (batch_size, num_heads, head_dim)
        Returns:
            out: Classification logits
            selection: Head selection probabilities
        """        

        print(heads_tensor.shape)
        head_encoders = torch.stack([encoder(heads_tensor[:, i, :]) for i, encoder in enumerate(self.head_encoders)], dim=1)
        attrs_encoders = torch.stack([encoder(attr_tensor[:, i, :]) for i, encoder in enumerate(self.head_encoders)], dim=1)

        head_encoders = self.norm_inputs(head_encoders)
        attrs_encoders = self.norm_inputs(attrs_encoders)

        pre_slot_output, pre_attr_output, attn_weights = self.head_grouper(head_encoders, attrs_encoders, 1)

        slot_output = torch.stack([proj(pre_slot_output[:, i]) for i, proj in enumerate(self.shared_proj) ], dim=1)
        attr_output = torch.stack([proj(pre_attr_output[:, i]) for i, proj in enumerate(self.shared_proj) ], dim=1)

        slot_output = self.norm_slots(slot_output)
        attr_output = self.norm_slots(attr_output)

        attr_prototype_dist = self.prototypes(attr_output)
        pers_prototype_dist = self.prototypes(slot_output)

        return pre_slot_output, pre_attr_output, pers_prototype_dist, attr_prototype_dist, attn_weights

@torch.no_grad()
def distributed_sinkhorn(out):
    # Implementation from: https://github.com/facebookresearch/swav
    Q = torch.exp(out / 0.5).t()
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q
    for it in range(3):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the columns must sum to 1 so that Q is an assignment
    return Q.t()