# model.py 
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import utils
from torch.utils.checkpoint import checkpoint

def sinusoidal_encoding(length, dim, device, dtype=torch.float32):
    """按需生成任意长度正弦位置编码 [length, dim]（无参数、无长度上限）"""
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    half = dim // 2
    inv = torch.exp(torch.arange(half, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / max(half, 1)))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * inv)
    pe[:, 1:2 * half:2] = torch.cos(pos * inv)
    return pe.to(dtype)


def split_bold_pe(features, pe_dim):
    """约定：.pt 中 features = cat([BOLD[T], LPE[pe]], dim=1)（PE 在最后 pe_dim 列）。
    变长后无法再靠 input_dim-pe_dim 反推，读取时必须显式拆开。"""
    return features[:, :-pe_dim].contiguous(), features[:, -pe_dim:].contiguous()


class TemporalTokenizer(nn.Module):
    """
    真变长 hop-token 编码器（线性时间池化版，保留 FC 结构）：

        tokens: [n, T+pe]
          -> BOLD 部分 [n, T] -> adaptive_avg_pool1d -> [n, B]
          -> Linear(B, H) -> [n, H]
          -> + pe_proj(LPE) -> [n, H]

    关键性质：
      - 整个映射关于 BOLD 时间维是线性的（池化 + 线性），
        保留节点间的时间相关性（FC），不会像 GELU + softmax 池化那样抹掉。
      - T 任意长，仅要求 T >= 1；若 T < n_time_basis，用线性插值补齐。
      - chunk + checkpoint：峰值激活只与 chunk_size 相关，与 n_tokens、T 解耦。
    """
    def __init__(self, hidden_dim, pe_dim, n_time_basis=64,
                 chunk_size=16384, grad_ckpt=True):
        super().__init__()
        self.pe_dim = pe_dim
        self.n_time_basis = int(n_time_basis)
        self.chunk_size, self.grad_ckpt = chunk_size, grad_ckpt

        self.time_proj = nn.Linear(self.n_time_basis, hidden_dim)
        self.pe_proj = nn.Linear(pe_dim, hidden_dim)

    def _encode_chunk(self, ts):                              # ts: [c, 1, T]
        T = ts.size(-1)
        B = min(self.n_time_basis, T)
        x = Fn.adaptive_avg_pool1d(ts, B).squeeze(1)          # [c, B]
        if B < self.n_time_basis:
            # T < n_time_basis：线性插值到 n_time_basis，仍是线性操作
            x = Fn.interpolate(
                x.unsqueeze(1), size=self.n_time_basis,
                mode="linear", align_corners=False,
            ).squeeze(1)
        return self.time_proj(x)                              # [c, H]

    def forward(self, tokens):                                # tokens: [n, T+pe]
        T = tokens.size(1) - self.pe_dim
        ts_all, pe_all = tokens[:, :T].unsqueeze(1), tokens[:, T:]
        outs = []
        for i in range(0, tokens.size(0), self.chunk_size):
            ts, pe = ts_all[i:i + self.chunk_size], pe_all[i:i + self.chunk_size]
            h = (checkpoint(self._encode_chunk, ts, use_reentrant=False)
                 if (self.grad_ckpt and self.training) else self._encode_chunk(ts))
            outs.append(h + self.pe_proj(pe))
        return torch.cat(outs, dim=0)                         # [n, H]


class VarLenBrainTransformer(nn.Module):
    """保留 fast_re_features 的变长版。仅替换两处固定维度接口，其余复用原实现。"""

    def __init__(self, hops, n_class, pe_dim, hidden_dim=512, ffn_dim=64,
                 num_heads=8, n_layers=1, dropout_rate=0.1,
                 attention_dropout_rate=0.1, ssl_dim=128, mask_ratio=0.25,
                 decoder_layers=2, dec_channels=128, use_grad_ckpt=True,
                 n_time_basis=64, **kw):
        super().__init__()
        self.seq_len = hops + 1
        self.pe_dim, self.mask_ratio = pe_dim, mask_ratio
        self.use_grad_ckpt = use_grad_ckpt

        # —— 变长接口 1 ——
        self.bold_tokenizer = TemporalTokenizer(
            hidden_dim, pe_dim,
            n_time_basis=n_time_basis,
            grad_ckpt=use_grad_ckpt,
        )
        self.hop_emb = nn.Parameter(torch.zeros(self.seq_len, hidden_dim))  # 零初始化=初始等价原"nope"
        self.mask_value = nn.Parameter(torch.zeros(1))  # 标量广播；no_grad 传播下无梯度（同原 mask_token 现状）

        # —— 与原模型相同 ——
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
            for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.attn_layer = nn.Linear(2 * hidden_dim, 1)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.Linear1 = nn.Linear(hidden_dim // 2, n_class)
        self.ssl_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, ssl_dim))
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, dropout_rate, attention_dropout_rate, ffn_dim)
            for _ in range(decoder_layers)])
        self.decoder_ln = nn.LayerNorm(hidden_dim)

        # —— 变长接口 2：时间重建头（替代 decoder_pred = Linear(H, bold_dim)）——
        self.dec_c = dec_channels        # 时间基函数数量，欠拟合时 128->256
        self.dec_fc = nn.Linear(hidden_dim, self.dec_c)
        self.dec_up = nn.ConvTranspose1d(self.dec_c, 1, kernel_size=8, stride=8)

    # ---------- 复用组件 ----------
    def _readout(self, enc_seq):
        target = enc_seq[:, 0, :].unsqueeze(1).repeat(1, self.seq_len - 1, 1)
        node_t, nb_t = enc_seq[:, :1, :], enc_seq[:, 1:, :]
        a = Fn.softmax(self.attn_layer(torch.cat((target, nb_t), dim=2)), dim=1)
        return (node_t + (nb_t * a).sum(dim=1, keepdim=True)).squeeze(1)

    def _run_encoder(self, x):
        for enc in self.layers:
            x = (checkpoint(enc, x, use_reentrant=False)
                 if (self.use_grad_ckpt and self.training) else enc(x))
        return self.final_ln(x)

    def _tokenize_hops(self, hop_rows):          # [n, K+1, F] -> [n, K+1, H]
        n, Kp1, F_ = hop_rows.shape
        tok = self.bold_tokenizer(hop_rows.reshape(n * Kp1, F_))
        return tok.view(n, Kp1, -1) + self.hop_emb[None]

    def decode_time(self, h, T):                 # [M,H] -> [M,T]，任意 T
        Tp = math.ceil(T / 8)
        u = self.dec_fc(h).unsqueeze(-1)                                         # [M,C,1]
        u = u + sinusoidal_encoding(Tp, self.dec_c, h.device, h.dtype).T[None]   # [M,C,Tp]
        return self.dec_up(u)[..., :T].squeeze(1)                                # [M,T]

    # ---------- 下游 / SSL（推理请在外层包 no_grad）----------
    def extract_node_representation(self, adj, X_raw, pe_feat):
        features = torch.cat([X_raw, pe_feat], dim=1)                            # [N, T+pe]
        hop_sequence = utils.fast_re_features(adj, features, K=self.seq_len - 1) # 原逻辑
        enc_seq = self._run_encoder(self._tokenize_hops(hop_sequence))
        return self._readout(enc_seq)

    def forward(self, adj, X_raw, pe_feat, return_representation=False, use_ssl_head=False):
        node_rep = self.extract_node_representation(adj, X_raw, pe_feat)
        if return_representation:
            return node_rep
        if use_ssl_head:
            return node_rep, self.ssl_head(node_rep)
        return torch.log_softmax(self.Linear1(torch.relu(self.out_proj(node_rep))), dim=1)

    # ---------- MAE 预训练 ----------
    def forward_mae(self, adj, X_raw, pe_feat):
        N, T = X_raw.shape
        M = max(1, int(N * self.mask_ratio))
        mask_idx = torch.randperm(N, device=X_raw.device)[:M]

        # 1) 只遮 BOLD、保留 PE（与原语义一致）
        X_masked = X_raw.clone()
        X_masked[mask_idx] = self.mask_value

        # 2) 保留 fast_re_features：原始特征空间 K 跳传播（no_grad，同原实现）
        features = torch.cat([X_masked, pe_feat], dim=1)
        hop_sequence = utils.fast_re_features(adj, features, K=self.seq_len - 1)  # [N,K+1,F]

        # 可选：恢复 hop-0 梯度通路让 mask_value 可学习（前向数值不变，仅多一条梯度路径）
        # hop_sequence = hop_sequence.clone()
        # hop_sequence[:, 0, :] = features

        # 3) 只取 masked 行 —— 节点间已无耦合，与"全图算完再取 masked"的 loss 严格等价
        enc_in = self._tokenize_hops(hop_sequence[mask_idx])                      # [M,K+1,H]
        del hop_sequence, features, X_masked                                      # 及时释放 no_grad 大缓冲

        enc_seq = self._run_encoder(enc_in)
        z_v = self._readout(enc_seq)

        # 4) cross-attn decoder（原样，仅 masked）
        q = z_v.unsqueeze(1)
        for dec in self.decoder_layers:
            q = dec(q, enc_seq)
        h = self.decoder_ln(q).squeeze(1)

        # 5) 变长重建 + loss
        pred = self.decode_time(h, T)
        loss = Fn.mse_loss(pred.float(), X_raw[mask_idx].float())
        mask = torch.zeros(N, dtype=torch.bool, device=X_raw.device)
        mask[mask_idx] = True
        return loss, pred, mask

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


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

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

class DecoderLayer(nn.Module):
    """解码器层：交叉注意力 (query 来自聚合节点，key/value 来自序列) + FFN"""
    def __init__(self, hidden_dim, num_heads, dropout_rate, attention_dropout_rate, ffn_dim):
        super().__init__()
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = MultiHeadAttention(hidden_dim, attention_dropout_rate, num_heads)
        self.dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForwardNetwork(hidden_dim, ffn_dim, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key_value_seq):
        """
        query: [N, 1, H] (聚合后的节点表征，扩展为 1 个 token)
        key_value_seq: [N, seq_len, H] (编码器完整序列)
        """
        # Cross-Attention: Q=query, K=V=key_value_seq
        attn_out = self.cross_attn_norm(query)
        attn_out = self.cross_attn(attn_out, key_value_seq, key_value_seq)
        attn_out = self.dropout(attn_out)
        query = query + attn_out

        # FFN
        ffn_out = self.ffn_norm(query)
        ffn_out = self.ffn(ffn_out)
        ffn_out = self.ffn_dropout(ffn_out)
        query = query + ffn_out
        return query
