import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from .utils import transformer_FFN, ut_mask, pos_encode, get_clones
from .que_base_model import QueBaseModel
from pykt.utils import debug_print

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


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


class BAIMParallelMoE(nn.Module):
    def __init__(self, input_dim, model_dim, dropout=0.1, top_k=1, state_dim=64):
        super().__init__()
        self.num_stages = 4
        self.model_dim = model_dim
        self.top_k = top_k
        self.last_gate_weights = None

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

        self.router_input_adapter = nn.Sequential(
            nn.Linear(input_dim * self.num_stages, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.w_gate = nn.Linear(model_dim + state_dim, self.num_stages)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.simple_projector = nn.Linear(input_dim, model_dim)

    def get_smart_summary(self, stage_repr_seq):
        batch_size, seq_len, _, _ = stage_repr_seq.shape
        q_flattened = stage_repr_seq.view(batch_size, seq_len, -1)
        return self.router_input_adapter(q_flattened)

    def forward(self, stage_repr_seq, m_seq, has_stages):
        if not has_stages:
            self.last_gate_weights = None
            return self.simple_projector(stage_repr_seq), None

        expert_outputs = []
        for k in range(self.num_stages):
            out = self.experts[k](stage_repr_seq[:, :, k, :])
            expert_outputs.append(out)
        experts_stacked = torch.stack(expert_outputs, dim=2)

        s_t = self.get_smart_summary(stage_repr_seq)
        router_input = torch.cat([s_t, m_seq], dim=-1)

        gate_logits = self.w_gate(router_input)

        if self.training:
            noise = torch.randn_like(gate_logits) * (1.0 / self.num_stages)
            noisy_logits = gate_logits + noise
        else:
            noisy_logits = gate_logits

        gate_probs = F.softmax(noisy_logits, dim=-1)

        top_k_vals, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)
        mask = torch.full_like(noisy_logits, float("-inf"))
        mask.scatter_(2, top_k_indices, top_k_vals)
        gate_weights = F.softmax(mask, dim=-1)

        if not self.training:
            self.last_gate_weights = gate_weights.detach().cpu()

        combined_q = (experts_stacked * gate_weights.unsqueeze(-1)).sum(dim=2)

        return self.layer_norm(combined_q), gate_probs


class SparseKTBAIMInteractionEmbedder(nn.Module):

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

        self.state_dim = 64
        self.input_reducer = nn.Linear(model_emb_size + model_emb_size, self.state_dim)
        self.state_gru = nn.GRU(self.state_dim, self.state_dim, batch_first=True)
        self.start_token_state = nn.Parameter(torch.zeros(1, 1, self.state_dim))

        self.polya_projector = BAIMParallelMoE(
            raw_emb_size, model_emb_size, dropout, top_k=1, state_dim=self.state_dim
        )

        self.response_emb = nn.Embedding(2, model_emb_size)

        debug_print(
            f"SparseKTBAIMInteractionEmbedder initialized.",
            fuc_name=model_name,
        )

    def forward(self, q, r):
        batch_size, seq_len = q.shape

        stage_repr_seq, has_stages = self.que_emb(q)
        r_emb = self.response_emb(r)

        if has_stages:
            s_t = self.polya_projector.get_smart_summary(stage_repr_seq)
        else:
            s_t = self.polya_projector.simple_projector(stage_repr_seq)

        interaction = torch.cat([s_t, r_emb], dim=-1)
        gru_input_seq = self.input_reducer(interaction)

        gru_input_shifted = torch.cat(
            [
                self.start_token_state.expand(batch_size, -1, -1),
                gru_input_seq[:, :-1, :],
            ],
            dim=1,
        )

        m_seq, _ = self.state_gru(gru_input_shifted)

        I_t, p_t = self.polya_projector(stage_repr_seq, m_seq, has_stages)
        self.last_gate_probs = p_t

        return I_t

    def get_attention_weights(self):
        return self.polya_projector.last_gate_weights


class SparseKTBAIMNet(nn.Module):
    def __init__(
        self,
        n_question,
        n_pid,
        d_model,
        n_blocks,
        dropout,
        d_ff=256,
        loss1=0.5,
        loss2=0.5,
        loss3=0.5,
        start=50,
        num_layers=2,
        nheads=4,
        seq_len=200,
        kq_same=1,
        final_fc_dim=512,
        final_fc_dim2=256,
        num_attn_heads=8,
        separate_qa=False,
        l2=1e-5,
        emb_type="qid",
        emb_path="",
        flag_load_emb=False,
        flag_emb_freezed=False,
        pretrain_dim=768,
        sparse_ratio=0.8,
        k_index=5,
        stride=1,
    ):
        super().__init__()

        self.model_name = "sparsekt_baim"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type

        self.sparse_ratio = sparse_ratio
        self.k_index = k_index
        self.stride = stride

        raw_emb_size = pretrain_dim if flag_load_emb else d_model
        self.polya_emb = SparseKTBAIMInteractionEmbedder(
            n_question,
            raw_emb_size,
            d_model,
            dropout,
            emb_path,
            flag_load_emb,
            flag_emb_freezed,
            self.model_name,
        )

        embed_l = d_model
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)

        self.model = Architecture(
            n_question=n_question,
            n_blocks=n_blocks,
            n_heads=num_attn_heads,
            dropout=dropout,
            d_model=d_model,
            d_feature=d_model // num_attn_heads,
            d_ff=d_ff,
            kq_same=self.kq_same,
            model_type="sparsekt_que",
            seq_len=seq_len,
        )

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1),
        )

    def forward(
        self,
        pid_data,
        q_data,
        target,
        attn_grads=None,
        save_path="",
        save_attn_path="",
        save_grad_path="",
        attn_cnt_path="",
    ):
        q_fused = self.polya_emb(q_data, target)

        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_fused

        d_output, attn_weights = self.model(
            q_embed_data=q_fused,
            qa_embed_data=qa_embed_data,
            emb_type=self.emb_type,
            sparse_ratio=self.sparse_ratio,
            k_index=self.k_index,
            attn_grads=attn_grads,
            stride=self.stride,
            save_path=save_path,
            save_attn_path=save_attn_path,
            save_grad_path=save_grad_path,
            attn_cnt_path=attn_cnt_path,
            q_data=q_data,
            n_question=self.n_question,
        )

        concat_q = torch.cat([d_output, q_fused], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)

        return preds, attn_weights

    def get_attention_weights(self):
        return self.polya_emb.get_attention_weights()


class SparseKTBAIM(QueBaseModel):
    def __init__(
        self,
        n_question,
        n_pid,
        d_model,
        n_blocks,
        dropout,
        d_ff=256,
        loss1=0.5,
        loss2=0.5,
        loss3=0.5,
        start=50,
        num_layers=2,
        nheads=4,
        seq_len=200,
        kq_same=1,
        final_fc_dim=512,
        final_fc_dim2=256,
        num_attn_heads=8,
        separate_qa=False,
        l2=1e-5,
        emb_type="qid",
        emb_path="",
        flag_load_emb=False,
        flag_emb_freezed=False,
        pretrain_dim=768,
        device="cpu",
        seed=0,
        sparse_ratio=0.8,
        k_index=5,
        stride=1,
        **kwargs,
    ):

        model_name = "sparsekt_baim"
        super().__init__(
            model_name=model_name,
            emb_type=emb_type,
            emb_path=emb_path,
            pretrain_dim=pretrain_dim,
            device=device,
            seed=seed,
        )

        debug_print(
            f"Initializing SparseKTBAIM (State-Aware MoE + SparseKT + Balancing)...",
            fuc_name=model_name,
        )

        self.model = SparseKTBAIMNet(
            n_question=n_question,
            n_pid=n_pid,
            d_model=d_model,
            n_blocks=n_blocks,
            dropout=dropout,
            d_ff=d_ff,
            loss1=loss1,
            loss2=loss2,
            loss3=loss3,
            start=start,
            num_layers=num_layers,
            nheads=nheads,
            seq_len=seq_len,
            kq_same=kq_same,
            final_fc_dim=final_fc_dim,
            final_fc_dim2=final_fc_dim2,
            num_attn_heads=num_attn_heads,
            separate_qa=separate_qa,
            l2=l2,
            emb_type=emb_type,
            emb_path=emb_path,
            flag_load_emb=flag_load_emb,
            flag_emb_freezed=flag_emb_freezed,
            pretrain_dim=pretrain_dim,
            sparse_ratio=sparse_ratio,
            k_index=k_index,
            stride=stride,
        )

        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")
        self.attn_weights = None

    def get_load_balancing_loss(self, gate_probs):
        if gate_probs is None:
            return 0.0
        num_experts = gate_probs.size(-1)
        gate_probs = gate_probs.view(-1, num_experts)
        expert_usage = gate_probs.mean(dim=0)
        target_usage = 1.0 / num_experts
        balance_loss = torch.sum((expert_usage - target_usage) ** 2)
        return balance_loss

    def train_one_step(self, data, process=True, return_all=False, weighted_loss=0):
        outputs, data_new = self.predict_one_step(
            data, return_details=True, process=process
        )

        main_loss = self.get_loss(
            outputs["y"], data_new["rshft"], data_new["sm"], weighted_loss=weighted_loss
        )

        gate_probs = self.model.polya_emb.last_gate_probs
        if gate_probs is not None:
            aux_loss = self.get_load_balancing_loss(gate_probs)
            total_loss = main_loss + (0.01 * aux_loss)
        else:
            total_loss = main_loss

        return outputs["y"], total_loss

    def predict_one_step(
        self, data, return_details=False, process=True, return_raw=False
    ):
        data_new = self.batch_to_device(data, process=process)
        y, attn_weights = self.model(
            data_new["cq"].long(), data_new["cq"].long(), data_new["cr"].long()
        )

        self.attn_weights = attn_weights
        outputs = {"y": y[:, 1:]}

        if return_details:
            return outputs, data_new
        else:
            return outputs["y"]

    def get_attention_weights(self):
        return self.model.get_attention_weights()


class Architecture(nn.Module):
    def __init__(
        self,
        n_question,
        n_blocks,
        d_model,
        d_feature,
        d_ff,
        n_heads,
        dropout,
        kq_same,
        model_type,
        seq_len,
    ):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type

        # SparseKT uses blocks_2
        self.blocks_2 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same,
                )
                for _ in range(n_blocks)
            ]
        )
        self.position_emb = CosinePositionalEmbedding(
            d_model=self.d_model, max_len=seq_len
        )

    def forward(
        self,
        q_embed_data,
        qa_embed_data,
        emb_type="qid",
        sparse_ratio=0.8,
        k_index=5,
        attn_grads=None,
        stride=1,
        save_path="",
        save_attn_path="",
        save_grad_path="",
        attn_cnt_path="",
        q_data=None,
        n_question=None,
    ):

        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        y = qa_embed_data
        x = q_embed_data

        for block in self.blocks_2:
            x, attn_weights = block(
                mask=0,
                query=x,
                key=x,
                values=y,
                apply_pos=True,
                emb_type=emb_type,
                sparse_ratio=sparse_ratio,
                k_index=k_index,
                attn_grads=attn_grads,
                stride=stride,
                save_path=save_path,
                save_attn_path=save_attn_path,
                save_grad_path=save_grad_path,
                attn_cnt_path=attn_cnt_path,
                q_data=q_data,
                n_question=n_question,
            )
        return x, attn_weights


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

    def forward(
        self,
        mask,
        query,
        key,
        values,
        apply_pos=True,
        emb_type="qid",
        sparse_ratio=0.8,
        k_index=5,
        attn_grads=None,
        stride=1,
        save_path="",
        save_attn_path="",
        save_grad_path="",
        attn_cnt_path="",
        q_data=None,
        n_question=None,
    ):

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query.device)

        if mask == 0:
            query2, attn_weights = self.masked_attn_head(
                query,
                key,
                values,
                mask=src_mask,
                zero_pad=True,
                emb_type=emb_type,
                sparse_ratio=sparse_ratio,
                k_index=k_index,
                attn_grads=attn_grads,
                stride=stride,
                save_path=save_path,
                save_attn_path=save_attn_path,
                save_grad_path=save_grad_path,
                attn_cnt_path=attn_cnt_path,
                q_data=q_data,
                n_question=n_question,
            )
        else:
            query2, attn_weights = self.masked_attn_head(
                query,
                key,
                values,
                mask=src_mask,
                zero_pad=False,
                emb_type=emb_type,
                sparse_ratio=sparse_ratio,
                k_index=k_index,
                attn_grads=attn_grads,
                stride=stride,
                save_path=save_path,
                save_attn_path=save_attn_path,
                save_grad_path=save_grad_path,
                attn_cnt_path=attn_cnt_path,
                q_data=q_data,
                n_question=n_question,
            )

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query, attn_weights


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

    def forward(
        self,
        q,
        k,
        v,
        mask,
        zero_pad,
        emb_type="qid",
        sparse_ratio=0.8,
        k_index=5,
        attn_grads=None,
        stride=1,
        save_path="",
        save_attn_path="",
        save_grad_path="",
        attn_cnt_path="",
        q_data=None,
        n_question=None,
    ):

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

        # Call Sparse Attention
        scores, attn_weights = attention(
            q,
            k,
            v,
            self.d_k,
            mask,
            self.dropout,
            zero_pad,
            emb_type,
            sparse_ratio=sparse_ratio,
            k_index=k_index,
            attn_grads=attn_grads,
            stride=stride,
            save_path=save_path,
            save_attn_path=save_attn_path,
            save_grad_path=save_grad_path,
            attn_cnt_path=attn_cnt_path,
            q_data=q_data,
            n_question=n_question,
        )

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)

        return output, attn_weights


def attention(
    q,
    k,
    v,
    d_k,
    mask,
    dropout,
    zero_pad,
    emb_type="qid",
    sparse_ratio=0.8,
    k_index=5,
    attn_grads=None,
    stride=1,
    save_path="",
    save_attn_path="",
    save_grad_path="",
    attn_cnt_path="",
    q_data=None,
    n_question=None,
):

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    if emb_type.find("sparseattn") != -1:
        if k_index >= seqlen:
            scores = scores
        else:
            scores_a = scores[:, :, :k_index, :]
            scores_b = scores[:, :, k_index:, :].reshape(
                bs * head * (seqlen - k_index), -1
            )
            sorted_scores, sorted_idx = torch.sort(scores_b, descending=True)
            scores_t = sorted_scores[:, k_index - 1 : k_index].repeat(1, seqlen)

            scores_b = torch.where(
                scores_b - scores_t >= torch.tensor(0).to(q.device),
                scores_b,
                torch.tensor(-1e32).to(q.device),
            ).reshape(bs, head, seqlen - k_index, -1)
            scores = torch.cat([scores_a, scores_b], dim=2)
            scores = F.softmax(scores, dim=-1)

    elif emb_type.find("accumulative") != -1:
        scores_flat = torch.reshape(scores, (bs * head * seqlen, -1))
        sorted_scores, sorted_idx = torch.sort(scores_flat, descending=True)
        acc_scores = torch.cumsum(sorted_scores, dim=1)
        mask_indices = (acc_scores <= sparse_ratio).int()
        mask_indices_last = torch.zeros_like(mask_indices)
        mask_indices_last[:, :-1] = mask_indices[:, 1:]
        mask_indices = mask_indices - mask_indices_last

        sparse_mask = torch.zeros_like(scores_flat).scatter_(
            1, sorted_idx, mask_indices.float()
        )
        sparse_mask = torch.reshape(sparse_mask, (bs, head, seqlen, -1))
        scores = torch.where(sparse_mask > 0, scores, torch.tensor(-1e32).to(q.device))
        scores = F.softmax(scores, dim=-1)

    before_dropout_scores = scores

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(q.device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)

    scores = dropout(scores)
    output = torch.matmul(scores, v)

    return output, before_dropout_scores


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, : x.size(Dim.seq), :]
