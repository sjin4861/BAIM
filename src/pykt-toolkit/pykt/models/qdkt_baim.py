import torch
from torch import nn
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from .que_base_model import QueBaseModel
from pykt.utils import debug_print


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
        routing_input = torch.cat([s_t, m_seq], dim=-1)

        alpha_t = self.w_gate(routing_input)

        if self.training:
            noise = torch.randn_like(alpha_t) * (1.0 / self.num_stages)
            alpha_t_noisy = alpha_t + noise
        else:
            alpha_t_noisy = alpha_t

        p_t = F.softmax(alpha_t_noisy, dim=-1)

        top_k_vals, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)
        mask = torch.full_like(noisy_logits, float("-inf"))
        mask.scatter_(2, top_k_indices, top_k_vals)
        gate_weights = F.softmax(mask, dim=-1)

        if not self.training:
            self.last_gate_weights = routing_weights.detach().cpu()

        I_t = (experts_stacked * routing_weights.unsqueeze(-1)).sum(dim=2)

        return self.layer_norm(I_t), p_t


class QDKTBAIMInteractionEmbedder(nn.Module):
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

        self.projector_incorrect = nn.Linear(model_emb_size, model_emb_size)
        self.projector_correct = nn.Linear(model_emb_size, model_emb_size)
        self.response_emb = nn.Embedding(2, model_emb_size)

        debug_print(
            f"QDKTBAIMInteractionEmbedder (Smart Proxy + Parallel + Balancing) initialized.",
            fuc_name=model_name,
        )

    def forward(self, x):
        batch_size, seq_len = x.shape
        r_recovered = (x >= self.num_q).long()
        q_recovered = x % self.num_q

        stage_repr_seq, has_stages = self.que_emb(q_recovered)
        r_emb = self.response_emb(r_recovered)

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

        emb_incorrect = self.projector_incorrect(I_t)
        emb_correct = self.projector_correct(I_t)

        r_mask = r_recovered.unsqueeze(-1).float()
        final_emb = emb_correct * r_mask + emb_incorrect * (1 - r_mask)

        return final_emb

    def get_attention_weights(self):
        return self.polya_projector.last_gate_weights


class QDKTBAIMNet(nn.Module):
    def __init__(
        self,
        num_q,
        raw_emb_size,
        model_emb_size,
        dropout=0.1,
        emb_type="qaid",
        emb_path="",
        flag_load_emb=False,
        flag_emb_freezed=False,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.model_name = "qdkt_baim"
        self.num_q = num_q
        self.emb_size = model_emb_size
        self.hidden_size = model_emb_size
        self.device = device

        self.interaction_emb = QDKTBAIMInteractionEmbedder(
            num_q,
            raw_emb_size,
            model_emb_size,
            dropout,
            emb_path,
            flag_load_emb,
            flag_emb_freezed,
            self.model_name,
        )

        self.lstm_layer = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.out_layer = nn.Linear(self.hidden_size, self.num_q)

        debug_print(f"QDKTBAIMNet initialized.", fuc_name=self.model_name)

    def forward(self, q, c, r, data=None):
        x = (q + self.num_q * r)[:, :-1]
        xemb = self.interaction_emb(x)
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        y = (y * F.one_hot(data["qshft"].long(), self.num_q)).sum(-1)
        outputs = {"y": y}
        return outputs

    def get_attention_weights(self):
        return self.interaction_emb.get_attention_weights()


class QDKTBAIM(QueBaseModel):
    def __init__(
        self,
        num_q,
        emb_size=100,
        dropout=0.1,
        emb_type="qid",
        emb_path="",
        flag_load_emb=False,
        flag_emb_freezed=False,
        pretrain_dim=768,
        device="cpu",
        seed=0,
        **kwargs,
    ):

        model_name = "qdkt_baim"
        debug_print(f"Initializing QDKTBAIM (With Aux Loss)...", fuc_name=model_name)

        super().__init__(
            model_name=model_name,
            emb_type=emb_type,
            emb_path=emb_path,
            pretrain_dim=pretrain_dim,
            device=device,
            seed=seed,
        )

        raw_emb_size = pretrain_dim if flag_load_emb else emb_size
        model_emb_size = emb_size

        self.model = QDKTBAIMNet(
            num_q=num_q,
            raw_emb_size=raw_emb_size,
            model_emb_size=model_emb_size,
            dropout=dropout,
            emb_type=emb_type,
            emb_path=emb_path,
            flag_load_emb=flag_load_emb,
            flag_emb_freezed=flag_emb_freezed,
            device=device,
        )
        self.model = self.model.to(device)
        self.emb_type = emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")

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

        gate_probs = self.model.interaction_emb.last_gate_probs
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
        outputs = self.model(
            data_new["cq"].long(), data_new["cc"], data_new["cr"].long(), data=data_new
        )

        if return_details:
            return outputs, data_new
        else:
            return outputs["y"]

    def get_attention_weights(self):
        return self.model.get_attention_weights()
