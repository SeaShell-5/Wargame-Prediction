import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)，进行线性变换
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]，加权求和得到输出

        x = x.transpose(1, 2).contiguous()  
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate=0.0):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class EncoderLayer(torch.nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.self_ffn_norm = nn.LayerNorm(hidden_size)
        self.self_ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.self_ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.self_ffn_norm(x)
        y = self.self_ffn(y)
        y = self.self_ffn_dropout(y)
        x = x + y
        return x
       

class MultiVariateMultiStepTransformer(nn.Module):
    def __init__(self, input_size, en_hidden, de_hidden, num_layers, num_heads, input_steps, out_steps, num_variables):
        super(MultiVariateMultiStepTransformer, self).__init__()
        self.input_proj = nn.Linear(input_size, en_hidden)
        encoder_layer = TransformerEncoderLayer(d_model=en_hidden, nhead=num_heads)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        # self.decoder = nn.Linear(hidden_size * input_steps, num_variables * out_steps)
        self.decoder = nn.Linear(en_hidden * input_steps , de_hidden * out_steps)
        self.out_steps = out_steps
        self.input_steps = input_steps
        self.en_hidden = en_hidden
        self.de_hidden = de_hidden
        self.num_variables = num_variables
        # self.decoder_norm = nn.LayerNorm(num_variables * out_steps)
        self.decoder_norm = nn.LayerNorm(de_hidden * out_steps )

    def forward(self, src):
        # src: (batch_size, seq_len, num_variables)
        # print(src.shape)  
        x = self.input_proj(src)  # (batch_size, seq_len, hidden_size)  
        x = x.permute(1, 0, 2)    # (seq_len, batch_size, hidden_size)
        x = self.encoder(x)        # (seq_len, batch_size, hidden_size)
        x = x.transpose(0, 1).reshape(-1, self.input_steps * self.en_hidden)
        # print(x.shape)
        # x = x[-self.out_steps:].transpose(0, 1)    # 取最后一个时间步 (batch_size, self.input_steps * self.hidden_size)
        # x = F.relu(x.view(-1, self.out_steps, self.hidden_size))
        # print(x.shape)
        x = self.decoder(x)        # (batch_size, num_variables * num_steps)
        x = F.relu(self.decoder_norm(x))
        x = F.relu(x.view(-1, self.out_steps, self.de_hidden))  # (batch_size, num_steps, num_variables)
            
        return x