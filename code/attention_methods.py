import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython import embed

class CosBahAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CosBahAttention, self).__init__()
        self.cos = CosSimAttention(in_dim, out_dim)
        self.bah = BahdanauAttention(in_dim, out_dim)
        #self.scale_hcx = nn.Linear(in_dim, out_dim)
        self.linear_combine = nn.Linear(out_dim*2, out_dim)
    
    def forward(self, k, xs, mask):
        # self.scale_hcx(k)
        cos_repr, cos_atts = self.cos(k, xs, mask)
        bah_repr, bah_atts = self.bah(k, xs, mask)
        
        attn = self.linear_combine(torch.cat([cos_repr, bah_repr], -1))
        atts = (cos_atts + bah_atts)/2
        

        return attn, atts

class CosSimAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CosSimAttention, self).__init__()
        #self.softabs = lambda x, epsilon: torch.sqrt(torch.pow(x, 2.0) + epsilon)
        self.eps = 1e-3 # 1e-14
        self.cos = nn.CosineSimilarity(dim=-1)
    
    def forward(self, k, xs, mask):
        cos_sim = self.cos(k.unsqueeze(1).repeat(1,xs.size(1),1), xs)
        cos_sim.data.masked_fill_(~mask.bool(), float('-inf'))
        #a_cos = self.softabs(cos_sim, self.eps)# * mask + EPS
        a_cos = F.softmax(cos_sim, -1)
        attn = torch.einsum('bs,bsk->bk', [a_cos, xs])
        return attn, a_cos
    
class BahdanauAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BahdanauAttention, self).__init__()
        self.energy = nn.Parameter(torch.randn(out_dim))
        self.Wa = nn.Linear(in_dim, out_dim, bias=False)
        self.Wb = nn.Linear(out_dim, out_dim, bias=False)
        
    def forward(self, k, xs, mask):
        ks = k.unsqueeze(1).repeat(1,xs.size(1),1)
        w = torch.tanh(self.Wa(ks) + self.Wb(xs))
        # calculate "energy"
        w = w @ self.energy
        w.data.masked_fill_(~mask.bool(), float('-inf'))
        attn = F.softmax(w, -1)# * mask + EPS
        return torch.einsum('bi,bik->bk', [attn, xs]), attn
    
class BahdanauAttention2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BahdanauAttention2, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(in_dim))
        self.Wa = nn.Linear(in_dim, in_dim, bias=False)
        self.Wb = nn.Linear(in_dim, in_dim, bias=False)
        self.e = nn.Linear(in_dim, 1, bias=False)
        
    def forward(self, k, xs, mask):
        ks = k.unsqueeze(1).repeat(1,xs.size(1),1)
        w = torch.tanh(self.Wa(ks) + self.Wb(xs)) + self.bias.expand(xs.size(0), xs.size(1), -1)
        w = self.e(w).squeeze(-1)
        
        w.data.masked_fill_(~mask.bool(), float('-inf'))
        
        attn = F.softmax(w, -1)# * mask + EPS
        return torch.einsum('bi,bik->bk', [attn, xs])
    
class SelfAttentionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SelfAttentionHead, self).__init__()
        self.k = nn.Linear(in_dim, out_dim) # remove bias here
        self.q = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.lin = nn.Linear(out_dim, out_dim)
        
    def forward(self, xs, mask):
        qk = torch.einsum('bsd,bde->bse', [self.k(xs), self.q(xs).transpose(2,1)])
        mask = mask.repeat(1,qk.size(1)).view(qk.size())
        qk_scaled = qk/xs.size(-1)
        qk_scaled.data.masked_fill_(~mask.bool(), float('-inf'))
        a = F.softmax(qk_scaled, -1)# * mask + EPS
        attn = torch.einsum('bij,bik->bik', [a, self.v(xs)])
        attn = F.leaky_relu(self.lin(attn))
        return attn
    
class SelfAttentionEncoder(nn.Module):
    def __init__(self, dim):
        super(SelfAttentionEncoder, self).__init__()
        self.dim = dim
        self.k = nn.Linear(dim, dim)
        self.q = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.lin = nn.Linear(dim, dim)
        
    def forward(self, xs):
        qk = torch.einsum('bsd,bde->bse', [self.k(xs), self.q(xs).transpose(2,1)])
        a = F.softmax(qk/self.dim, 1)
        attn = torch.einsum('bij,bik->bik', [a, self.v(xs)])
        return torch.tanh(self.lin(attn))