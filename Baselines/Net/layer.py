from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F



# 动态的图卷积，对应论文中动态构建图结构
class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nvwl->ncwl', (x, A))
        return x.contiguous()


# 线性变换
class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


# class prop(nn.Module):
#     def __init__(self, c_in, c_out, gdep, dropout, alpha):
#         super(prop, self).__init__()
#         self.nconv = nconv()
#         self.mlp = linear(c_in, c_out)
#         self.gdep = gdep
#         self.dropout = dropout
#         self.alpha = alpha

#     def forward(self, x, adj):
#         adj = adj + torch.eye(adj.size(0)).to(x.device)
#         d = adj.sum(1)
#         h = x
#         dv = d
#         a = adj / dv.view(-1, 1)
#         for i in range(self.gdep):
#             h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
#         ho = self.mlp(h)
#         return ho


# # 静态图结构，需要传入固定的邻接矩阵
# class mixprop(nn.Module):
#     def __init__(self, c_in, c_out, gdep, dropout, alpha):
#         super(mixprop, self).__init__()
#         self.nconv = nconv()
#         self.mlp = linear((gdep + 1) * c_in, c_out)
#         self.gdep = gdep
#         self.dropout = dropout
#         self.alpha = alpha

#     def forward(self, x, adj):
#         adj = adj + torch.eye(adj.size(0)).to(x.device)
#         d = adj.sum(1)
#         h = x
#         out = [h]
#         a = adj / d.view(-1, 1)
#         for i in range(self.gdep):
#             h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
#             out.append(h)
#         ho = torch.cat(out, dim=1)
#         ho = self.mlp(ho)
#         return ho

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

# 动态图结构，对应论文中Mix-hop Attention Propagation Layer
class dy_mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.lin1 = linear(c_in, c_in)
        self.lin2 = linear(c_in, c_in)

        self.atten1 = ScaledDotProductAttention(d_k=(gdep + 1) * c_in)
        self.atten2 = ScaledDotProductAttention(d_k=(gdep + 1) * c_in)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


        self.linear_out1 = linear((gdep + 1) * c_in, c_out)
        self.linear_out2 = linear((gdep + 1) * c_in, c_out)

    def forward(self, x):
        # 检查输入
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("dy_mixprop input contains nan or inf values")
            
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        
        # 检查线性变换输出
        if torch.isnan(x1).any() or torch.isinf(x1).any():
            raise ValueError("dy_mixprop x1 contains nan or inf values")
        if torch.isnan(x2).any() or torch.isinf(x2).any():
            raise ValueError("dy_mixprop x2 contains nan or inf values")
            
        adj = self.nconv(x1.transpose(2, 1), x2)
        
        # 检查邻接矩阵
        if torch.isnan(adj).any() or torch.isinf(adj).any():
            raise ValueError("dy_mixprop adj contains nan or inf values")
            
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2, 1), dim=2)
        
        # 检查softmax输出
        if torch.isnan(adj0).any() or torch.isinf(adj0).any():
            raise ValueError("dy_mixprop adj0 contains nan or inf values")
        if torch.isnan(adj1).any() or torch.isinf(adj1).any():
            raise ValueError("dy_mixprop adj1 contains nan or inf values")

        h = x
        out = [h]
        # 这是Mix-hop中的模块进行gdep个层卷积处理
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj0)
            out.append(h)
        ho1 = torch.cat(out, dim=1)
        Q1, K1, V1 = ho1, ho1, ho1
        ho1_attn = self.atten1(Q1, K1, V1)
        ho1 = self.linear_out1(ho1_attn)
        
        # 检查第一个分支输出
        if torch.isnan(ho1).any() or torch.isinf(ho1).any():
            raise ValueError("dy_mixprop ho1 contains nan or inf values")

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho2 = torch.cat(out, dim=1)
        Q2, K2, V2 = ho2, ho2, ho2
        ho2_attn = self.atten2(Q2, K2, V2)
        ho2 = self.linear_out2(ho2_attn)
        
        # 检查第二个分支输出
        if torch.isnan(ho2).any() or torch.isinf(ho2).any():
            raise ValueError("dy_mixprop ho2 contains nan or inf values")

        result = ho1 + ho2
        
        # 检查最终输出
        if torch.isnan(result).any() or torch.isinf(result).any():
            raise ValueError("dy_mixprop result contains nan or inf values")

        return result



class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        self.tconv = nn.Conv2d(cin, cout, (1, 7), dilation=(1, dilation_factor))

    def forward(self, input):
        x = self.tconv(input)
        return x


# 时间卷积模块中Dilated inception Layer
class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=4):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [1, 3, 5, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
