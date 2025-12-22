import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from .que_base_model import QueBaseModel
from pykt.utils import debug_print


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


# ------------------------------------------------------------------------------
# 1. QueEmbedder (Shared)
# ------------------------------------------------------------------------------
class QueEmbedder(nn.Module):
    def __init__(
        self, num_q, emb_size, emb_path, flag_load_emb, flag_emb_freezed, model_name
    ):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.emb_path = emb_path.strip().strip('"').strip("'") if emb_path else ""
        self.flag_load_emb = flag_load_emb
        self.flag_emb_freezed = flag_emb_freezed
        self.model_name = model_name
        self.loaded_emb_dim = emb_size
        self.num_stages = 4
        self.has_stages = False
        self.init_embedding_layer()
        if self.loaded_emb_dim != self.emb_size:
            self.projection_layer = nn.Linear(self.loaded_emb_dim, self.emb_size)
        else:
            self.projection_layer = nn.Identity()

    def init_embedding_layer(self):
        if self.emb_path == "" or not self.flag_load_emb:
            debug_print(
                f"Standard Random Embeddings (No Stages).", fuc_name=self.model_name
            )
            self.que_emb = nn.Embedding(self.num_q, self.emb_size)
            self.has_stages = False
        elif self.flag_load_emb:
            debug_print(
                f"Loading embeddings from: {self.emb_path}", fuc_name=self.model_name
            )
            if self.emb_path.endswith(".pt"):
                precomputed_tensor = torch.load(self.emb_path, map_location="cpu")
            else:
                raise ValueError(f"Only .pt files supported.")
            if precomputed_tensor.dim() == 3:
                num_q_loaded, num_stages_loaded, self.loaded_emb_dim = (
                    precomputed_tensor.shape
                )
                flattened_tensor = precomputed_tensor.reshape(
                    num_q_loaded * self.num_stages, -1
                )
                freeze = True if self.flag_emb_freezed else False
                self.que_emb = nn.Embedding.from_pretrained(
                    flattened_tensor, freeze=freeze
                )
                self.has_stages = True
                debug_print(f"Loaded 4-Stage Embeddings.", fuc_name=self.model_name)
            elif precomputed_tensor.dim() == 2:
                self.que_emb = nn.Embedding.from_pretrained(
                    precomputed_tensor, freeze=self.flag_emb_freezed
                )
                self.has_stages = False
                self.loaded_emb_dim = precomputed_tensor.shape[1]

    def forward(self, q):
        if not self.has_stages:
            x = self.que_emb(q)
            x = self.projection_layer(x)
            return x, False
        base_indices = q.unsqueeze(-1) * self.num_stages
        offsets = torch.arange(self.num_stages, device=q.device).view(1, 1, -1)
        stage_indices = base_indices + offsets
        x = self.que_emb(stage_indices)
        x = self.projection_layer(x)
        return x, True


# ------------------------------------------------------------------------------
# 2. PolyaParallelMoE (Identical to qDKT V5)
# ------------------------------------------------------------------------------
class PolyaParallelMoE(nn.Module):
    """
    Fully Parallel MoE Router & Expert Fusion.
    Provides 'get_smart_summary' for state tracking alignment.
    Returns 'gate_probs' for load balancing loss.
    """

    def __init__(self, input_dim, model_dim, dropout=0.1, top_k=1, state_dim=64):
        super().__init__()
        self.num_stages = 4
        self.model_dim = model_dim
        self.top_k = top_k
        self.last_gate_weights = None

        # 1. Experts
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim, model_dim),
                )
                for _ in range(self.num_stages)
            ]
        )

        # 2. Router Input Adapter (Public)
        self.router_input_adapter = nn.Sequential(
            nn.Linear(input_dim * self.num_stages, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 3. Router Gate
        self.w_gate = nn.Linear(model_dim + state_dim, self.num_stages)

        self.layer_norm = nn.LayerNorm(model_dim)
        self.simple_projector = nn.Linear(input_dim, model_dim)

    def get_smart_summary(self, q_raw_seq):
        """Returns learnable summary of 4 stages: (B, S, D)"""
        batch_size, seq_len, _, _ = q_raw_seq.shape
        q_flattened = q_raw_seq.view(batch_size, seq_len, -1)
        return self.router_input_adapter(q_flattened)

    def forward(self, q_raw_seq, m_seq, has_stages):
        if not has_stages:
            self.last_gate_weights = None
            return self.simple_projector(q_raw_seq), None

        # [1] Precompute Experts (Parallel)
        expert_outputs = []
        for k in range(self.num_stages):
            out = self.experts[k](q_raw_seq[:, :, k, :])
            expert_outputs.append(out)
        experts_stacked = torch.stack(expert_outputs, dim=2)

        # [2] Prepare Router Input (Parallel)
        q_summary = self.get_smart_summary(q_raw_seq)
        router_input = torch.cat([q_summary, m_seq], dim=-1)

        # [3] Calculate Logits
        gate_logits = self.w_gate(router_input)

        if self.training:
            noise = torch.randn_like(gate_logits) * (1.0 / self.num_stages)
            noisy_logits = gate_logits + noise
        else:
            noisy_logits = gate_logits

        # For Load Balancing Loss
        gate_probs = F.softmax(noisy_logits, dim=-1)

        # [4] Top-1 Selection
        top_k_vals, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)
        mask = torch.full_like(noisy_logits, float("-inf"))
        mask.scatter_(2, top_k_indices, top_k_vals)
        gate_weights = F.softmax(mask, dim=-1)

        if not self.training:
            self.last_gate_weights = gate_weights.detach().cpu()

        # [5] Fusion
        combined_q = (experts_stacked * gate_weights.unsqueeze(-1)).sum(dim=2)

        return self.layer_norm(combined_q), gate_probs


# ------------------------------------------------------------------------------
# 3. PolyaAKTInteractionEmbedder (New for AKT)
# ------------------------------------------------------------------------------
class PolyaAKTInteractionEmbedder(nn.Module):
    """
    [Polya for AKT]
    Generates TWO embeddings:
    1. q_fused: The 'Query' for Retrieval (Current Problem)
    2. qa_fused: The 'Value' for Context (History)

    Uses Parallel State Tracking + Smart Proxy Sharing.
    """

    def __init__(
        self,
        num_q,
        raw_emb_size,
        model_emb_size,
        dropout,
        emb_path,
        flag_load_emb,
        flag_emb_freezed,
        model_name,
    ):
        super().__init__()
        self.num_q = num_q
        self.model_name = model_name
        self.model_emb_size = model_emb_size
        self.last_gate_probs = None

        self.que_emb = QueEmbedder(
            num_q, raw_emb_size, emb_path, flag_load_emb, flag_emb_freezed, model_name
        )

        # [State Tracker]
        self.state_dim = 64
        self.input_reducer = nn.Linear(model_emb_size + model_emb_size, self.state_dim)
        self.state_gru = nn.GRU(self.state_dim, self.state_dim, batch_first=True)
        self.start_token_state = nn.Parameter(torch.zeros(1, 1, self.state_dim))

        # [MoE Projector]
        self.polya_projector = PolyaParallelMoE(
            raw_emb_size, model_emb_size, dropout, top_k=1, state_dim=self.state_dim
        )

        # [Dual Projection for QA Embedding]
        # AKT normally uses q + r_emb.
        # We upgrade this to V5 style: separate projections for correct/incorrect.
        self.projector_incorrect = nn.Linear(model_emb_size, model_emb_size)
        self.projector_correct = nn.Linear(model_emb_size, model_emb_size)
        self.response_emb = nn.Embedding(2, model_emb_size)

        debug_print(
            f"PolyaAKTInteractionEmbedder (V5 Logic) initialized.", fuc_name=model_name
        )

    def forward(self, q, r):
        batch_size, seq_len = q.shape

        # [Step 1] Bulk Lookup
        q_raw_seq, has_stages = self.que_emb(q)
        r_emb_seq = self.response_emb(r)

        # [Step 2] Smart Proxy Generation (Parallel)
        if has_stages:
            q_proxy_seq = self.polya_projector.get_smart_summary(q_raw_seq)
        else:
            q_proxy_seq = self.polya_projector.simple_projector(q_raw_seq)

        # [Step 3] Parallel State Tracking
        # State depends on (Question_Summary + Response)
        interaction_seq = torch.cat([q_proxy_seq, r_emb_seq], dim=-1)
        gru_input_seq = self.input_reducer(interaction_seq)

        # Shift Right (m_t depends on history 0..t-1)
        gru_input_shifted = torch.cat(
            [
                self.start_token_state.expand(batch_size, -1, -1),
                gru_input_seq[:, :-1, :],
            ],
            dim=1,
        )

        m_seq, _ = self.state_gru(gru_input_shifted)  # (B, S, 64)

        # [Step 4] Run MoE (Parallel)
        # Generates the optimal Question Representation
        q_fused, gate_probs = self.polya_projector(q_raw_seq, m_seq, has_stages)
        self.last_gate_probs = gate_probs  # Store for Loss

        # [Step 5] Generate Interaction Embedding (History)
        # Dual Projection Logic: Separate spaces for correct/incorrect
        emb_incorrect = self.projector_incorrect(q_fused)
        emb_correct = self.projector_correct(q_fused)

        r_mask = r.unsqueeze(-1).float()
        qa_fused = emb_correct * r_mask + emb_incorrect * (1 - r_mask)

        return q_fused, qa_fused

    def get_attention_weights(self):
        return self.polya_projector.last_gate_weights


# ------------------------------------------------------------------------------
# 4. PolyaAKTNetV5 (Main Network)
# ------------------------------------------------------------------------------
class PolyaAKTNetV5(nn.Module):
    def __init__(
        self,
        num_q,
        num_c,
        emb_size,
        n_blocks,
        dropout,
        d_ff=256,
        kq_same=1,
        final_fc_dim=512,
        num_attn_heads=8,
        separate_qa=False,
        l2=1e-5,
        emb_type="qid",
        emb_path="",
        flag_load_emb=False,
        flag_emb_freezed=False,
        pretrain_dim=768,
    ):
        super().__init__()

        self.model_name = "polya_akt_v5"
        self.num_q = num_q
        self.emb_size = emb_size
        self.dropout = dropout

        # 1. Polya V5 Embedding Module
        raw_emb_size = pretrain_dim if flag_load_emb else emb_size
        self.polya_emb = PolyaAKTInteractionEmbedder(
            num_q,
            raw_emb_size,
            emb_size,
            dropout,
            emb_path,
            flag_load_emb,
            flag_emb_freezed,
            self.model_name,
        )

        # 2. AKT Transformer Architecture (Standard)
        self.model = Architecture(
            num_q=num_q,
            n_blocks=n_blocks,
            n_heads=num_attn_heads,
            dropout=dropout,
            d_model=emb_size,
            d_feature=emb_size // num_attn_heads,
            d_ff=d_ff,
            kq_same=kq_same,
            model_type="akt_que",
        )

        # 3. Prediction Head
        self.out = nn.Sequential(
            nn.Linear(emb_size + emb_size, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )

    def forward(self, q, c, r):
        # 1. Get Embeddings from Polya V5
        # q_fused: Contextual Question Embedding (for Query)
        # qa_fused: Contextual History Embedding (for Key/Value)
        q_fused, qa_fused = self.polya_emb(q, r)

        # 2. AKT Transformer
        # Encoder uses qa_fused (History), Decoder uses q_fused (Current Q)
        d_output = self.model(q_fused, qa_fused)

        # 3. Predict
        concat_q = torch.cat([d_output, q_fused], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)

        return preds, 0.0

    def get_attention_weights(self):
        return self.polya_emb.get_attention_weights()


# ------------------------------------------------------------------------------
# 5. PolyaAKTV5 (Wrapper with Aux Loss)
# ------------------------------------------------------------------------------
class PolyaAKTV5(QueBaseModel):
    def __init__(
        self,
        num_q,
        num_c,
        emb_size,
        n_blocks=1,
        dropout=0.1,
        emb_type="qid",
        kq_same=1,
        final_fc_dim=512,
        num_attn_heads=8,
        separate_qa=False,
        l2=1e-5,
        d_ff=256,
        emb_path="",
        flag_load_emb=False,
        flag_emb_freezed=False,
        pretrain_dim=768,
        device="cpu",
        seed=0,
        **kwargs,
    ):

        model_name = "polya_akt_v5"
        super().__init__(
            model_name=model_name,
            emb_type=emb_type,
            emb_path=emb_path,
            pretrain_dim=pretrain_dim,
            device=device,
            seed=seed,
        )

        debug_print(
            f"Initializing PolyaAKTV5 (State-Aware MoE + AKT + Balancing)...",
            fuc_name=model_name,
        )

        self.model = PolyaAKTNetV5(
            num_q=num_q,
            num_c=num_c,
            emb_size=emb_size,
            n_blocks=n_blocks,
            dropout=dropout,
            d_ff=d_ff,
            kq_same=kq_same,
            final_fc_dim=final_fc_dim,
            num_attn_heads=num_attn_heads,
            separate_qa=separate_qa,
            l2=l2,
            emb_type=emb_type,
            emb_path=emb_path,
            flag_load_emb=flag_load_emb,
            flag_emb_freezed=flag_emb_freezed,
            pretrain_dim=pretrain_dim,
        )
        self.model = self.model.to(device)
        self.emb_type = self.model.model_name
        self.loss_func = self._get_loss_func("binary_crossentropy")

    def get_load_balancing_loss(self, gate_probs):
        """Calculates auxiliary loss for balanced expert usage."""
        if gate_probs is None:
            return 0.0
        num_experts = gate_probs.size(-1)
        gate_probs = gate_probs.view(-1, num_experts)
        expert_usage = gate_probs.mean(dim=0)
        target_usage = 1.0 / num_experts
        balance_loss = torch.sum((expert_usage - target_usage) ** 2)
        return balance_loss

    def train_one_step(self, data, process=True, weighted_loss=0):
        # AKT predicts next item (shifted output handled inside model or here)
        # Note: AKTQue usually outputs sequence 0..T, we match with rshft
        y, reg_loss, data_new = self.predict_one_step(
            data, return_details=True, process=process
        )

        # 1. Main Loss
        main_loss = self.get_loss(
            y, data_new["rshft"], data_new["sm"], weighted_loss=weighted_loss
        )

        # 2. Aux Loss (Load Balancing)
        gate_probs = self.model.polya_emb.last_gate_probs
        if gate_probs is not None:
            aux_loss = self.get_load_balancing_loss(gate_probs)
            total_loss = main_loss + (0.01 * aux_loss)  # lambda = 0.01
        else:
            total_loss = main_loss

        return y, total_loss

    def predict_one_step(self, data, return_details=False, process=True):
        data_new = self.batch_to_device(data, process=process)
        y, reg_loss = self.model(
            data_new["cq"].long(), data_new["cc"].long(), data_new["cr"].long()
        )

        # AKTQue convention: Output is aligned to next step prediction
        # Usually needs slicing to match ground truth length
        y = y[:, 1:]

        if return_details:
            return y, reg_loss, data_new
        else:
            return y

    def get_attention_weights(self):
        return self.model.get_attention_weights()


# ------------------------------------------------------------------------------
# 6. AKT Components (Standard Transformer)
# ------------------------------------------------------------------------------
class Architecture(nn.Module):
    def __init__(
        self,
        num_q,
        n_blocks,
        d_model,
        d_feature,
        d_ff,
        n_heads,
        dropout,
        kq_same,
        model_type,
    ):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type

        self.blocks_1 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_feature,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same,
                )
                for _ in range(n_blocks)
            ]
        )
        self.blocks_2 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_feature,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same,
                )
                for _ in range(n_blocks * 2)
            ]
        )

    def forward(self, q_embed_data, qa_embed_data):
        # Encoder (Encode History)
        y = qa_embed_data
        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y)  # yt^

        # Decoder (Retrieve Knowledge based on Question)
        x = q_embed_data
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question only
                x = block(mask=1, query=x, key=x, values=x, apply_pos=False)
                flag_first = False
            else:  # retrieve from history y
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        kq_same = kq_same == 1
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query.device)

        if mask == 0:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True
            )
        else:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False
            )

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)
        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        gammas = self.gammas
        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad, gammas)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)
        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(q.device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(q.device)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = (
            torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(q.device)
        )
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.0
        )
        dist_scores = dist_scores.sqrt().detach()

    m = nn.Softplus()
    gamma = -1.0 * m(gamma).unsqueeze(0)
    total_effect = torch.clamp(
        torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5
    )
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(q.device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)

    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output
