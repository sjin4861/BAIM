import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .que_base_model import QueBaseModel
from pykt.utils import debug_print
from sklearn import metrics
from torch.utils.data import DataLoader


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

    def get_smart_summary(self, q_raw_seq):
        batch_size, seq_len, _, _ = q_raw_seq.shape
        q_flattened = q_raw_seq.view(batch_size, seq_len, -1)
        return self.router_input_adapter(q_flattened)

    def forward(self, q_raw_seq, m_seq, has_stages):
        if not has_stages:
            self.last_gate_weights = None
            return self.simple_projector(q_raw_seq), None

        expert_outputs = []
        for k in range(self.num_stages):
            out = self.experts[k](q_raw_seq[:, :, k, :])
            expert_outputs.append(out)
        experts_stacked = torch.stack(expert_outputs, dim=2)

        q_summary = self.get_smart_summary(q_raw_seq)
        router_input = torch.cat([q_summary, m_seq], dim=-1)

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


# ------------------------------------------------------------------------------
# 3. PolyaQIKTInteractionEmbedder (New for QIKT)
# ------------------------------------------------------------------------------
class PolyaQIKTInteractionEmbedder(nn.Module):
    """
    [Polya for QIKT]
    Generates:
    1. q_fused: Polya-fused Question Embedding
    2. gate_probs: For Aux Loss

    Uses Parallel State Tracking + Smart Proxy Sharing.
    """

    def __init__(
        self,
        num_q,
        num_c,
        raw_emb_size,
        model_emb_size,
        dropout,
        emb_path,
        flag_load_emb,
        flag_emb_freezed,
        model_name,
        device,
    ):
        super().__init__()
        self.num_q = num_q
        self.num_c = num_c
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

        # Response Embedding (Used for State Tracking)
        self.response_emb = nn.Embedding(2, model_emb_size)

        # Concept Embeddings (QIKT Specific)
        self.concept_emb = nn.Parameter(
            torch.randn(self.num_c, self.model_emb_size).to(device), requires_grad=True
        )

        # Linear layer for combining question and concept (QIKT Logic)
        self.que_c_linear = nn.Linear(2 * self.model_emb_size, self.model_emb_size)

        debug_print(
            f"PolyaQIKTInteractionEmbedder (V5 Logic) initialized.", fuc_name=model_name
        )

    def get_avg_skill_emb(self, c):
        """Get average concept embedding (Identical to QIKT)"""
        # Add zero for padding
        concept_emb_cat = torch.cat(
            [torch.zeros(1, self.model_emb_size).to(c.device), self.concept_emb], dim=0
        )

        related_concepts = (c + 1).long()
        # [batch_size, seq_len, emb_dim]
        concept_emb_sum = concept_emb_cat[related_concepts, :].sum(axis=-2)

        # [batch_size, seq_len, 1]
        concept_num = (
            torch.where(related_concepts != 0, 1, 0)
            .sum(axis=-1)
            .unsqueeze(-1)
            .to(c.device)
        )
        concept_num = torch.where(concept_num == 0, 1, concept_num)
        concept_avg = concept_emb_sum / concept_num
        return concept_avg

    def forward(self, q, c, r):
        batch_size, seq_len = q.shape

        # [Step 1] Bulk Lookup
        q_raw_seq, has_stages = self.que_emb(q)
        r_emb_seq = self.response_emb(r)

        # [Step 2] Smart Proxy Generation
        if has_stages:
            q_proxy_seq = self.polya_projector.get_smart_summary(q_raw_seq)
        else:
            q_proxy_seq = self.polya_projector.simple_projector(q_raw_seq)

        # [Step 3] Parallel State Tracking
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

        m_seq, _ = self.state_gru(gru_input_shifted)

        # [Step 4] Run MoE (Parallel)
        q_fused, gate_probs = self.polya_projector(q_raw_seq, m_seq, has_stages)
        self.last_gate_probs = gate_probs  # Store for Loss

        # [Step 5] QIKT Specific Combining Logic
        emb_c = self.get_avg_skill_emb(c)  # [B, S, D]

        # Combine Q and C
        emb_qc = torch.cat([q_fused, emb_c], dim=-1)  # [B, S, 2D]
        xemb = self.que_c_linear(emb_qc)

        # Add Response Info (Interaction)
        # emb_qca: Interaction representation (correct vs incorrect)
        emb_qca = torch.cat(
            [
                emb_qc.mul((1 - r).unsqueeze(-1).repeat(1, 1, self.model_emb_size * 2)),
                emb_qc.mul((r).unsqueeze(-1).repeat(1, 1, self.model_emb_size * 2)),
            ],
            dim=-1,
        )

        # Return all components needed by QIKT
        return xemb, emb_qca, emb_qc, q_fused, emb_c

    def get_attention_weights(self):
        return self.polya_projector.last_gate_weights


# ------------------------------------------------------------------------------
# 4. MLP Classifier (Shared)
# ------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()
        self.lins = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layer)]
        )
        self.dropout = nn.Dropout(p=dpo)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))


# ------------------------------------------------------------------------------
# 5. PolyaQIKTNetV5 (Main Network)
# ------------------------------------------------------------------------------
class PolyaQIKTNetV5(nn.Module):
    def __init__(
        self,
        num_q,
        num_c,
        raw_emb_size,
        model_emb_size,
        dropout=0.1,
        emb_type="qaid",
        emb_path="",
        flag_load_emb=False,
        flag_emb_freezed=False,
        pretrain_dim=768,
        device="cpu",
        mlp_layer_num=1,
        other_config={},
    ):
        super().__init__()
        self.model_name = "polya_qikt_v5"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = model_emb_size
        self.hidden_size = model_emb_size
        self.mlp_layer_num = mlp_layer_num
        self.device = device
        self.other_config = other_config
        self.output_mode = self.other_config.get("output_mode", "an")
        self.emb_type = emb_type

        # 1. Embedder (Polya V5)
        self.polya_emb = PolyaQIKTInteractionEmbedder(
            num_q=num_q,
            num_c=num_c,
            raw_emb_size=raw_emb_size,
            model_emb_size=model_emb_size,
            dropout=dropout,
            emb_path=emb_path,
            flag_load_emb=flag_load_emb,
            flag_emb_freezed=flag_emb_freezed,
            model_name=self.model_name,
            device=device,
        )

        # 2. LSTMs (Identical to QIKTQue)
        self.que_lstm_layer = nn.LSTM(
            self.emb_size * 4, self.hidden_size, batch_first=True
        )
        self.concept_lstm_layer = nn.LSTM(
            self.emb_size * 2, self.hidden_size, batch_first=True
        )

        self.dropout_layer = nn.Dropout(dropout)

        # 3. Output MLPs
        self.out_question_next = MLP(
            self.mlp_layer_num, self.hidden_size * 3, 1, dropout
        )
        self.out_question_all = MLP(
            self.mlp_layer_num, self.hidden_size, num_q, dropout
        )
        self.out_concept_next = MLP(
            self.mlp_layer_num, self.hidden_size * 3, num_c, dropout
        )
        self.out_concept_all = MLP(self.mlp_layer_num, self.hidden_size, num_c, dropout)

    def get_avg_fusion_concepts(self, y_concept, cshft):
        """Get concept fusion prediction results"""
        max_num_concept = cshft.shape[-1]
        concept_mask = torch.where(cshft.long() == -1, False, True)
        concept_index = F.one_hot(torch.where(cshft != -1, cshft, 0).long(), self.num_c)
        concept_sum = (
            y_concept.unsqueeze(2).repeat(1, 1, max_num_concept, 1) * concept_index
        ).sum(-1)
        concept_sum = concept_sum * concept_mask  # Remove mask
        y_concept = concept_sum.sum(-1) / torch.where(
            concept_mask.sum(-1) != 0, concept_mask.sum(-1), 1
        )
        return y_concept

    def get_outputs(self, emb_qc_shift, h, data, add_name="", model_type="question"):
        """Get predictions for question or concept model"""
        outputs = {}
        h_next = torch.cat([emb_qc_shift, h], axis=-1)

        if model_type == "question":
            y_question_next = torch.sigmoid(self.out_question_next(h_next))
            y_question_all = torch.sigmoid(self.out_question_all(h))
            outputs["y_question_next" + add_name] = y_question_next.squeeze(-1)
            outputs["y_question_all" + add_name] = (
                y_question_all * F.one_hot(data["qshft"].long(), self.num_q)
            ).sum(-1)
        else:  # concept model
            y_concept_next = torch.sigmoid(self.out_concept_next(h_next))
            y_concept_all = torch.sigmoid(self.out_concept_all(h))
            outputs["y_concept_next" + add_name] = self.get_avg_fusion_concepts(
                y_concept_next, data["cshft"]
            )
            outputs["y_concept_all" + add_name] = self.get_avg_fusion_concepts(
                y_concept_all, data["cshft"]
            )
        return outputs

    def forward(self, q, c, r, data=None):
        # Get Embeddings (Polya Fused Q included)
        # q_fused is used inside emb_qc and emb_qca construction
        _, emb_qca, emb_qc, emb_q, emb_c = self.polya_emb(q, c, r)

        # Shift Data
        emb_qc_shift = emb_qc[:, 1:, :]  # Next Question Features (Target)
        emb_qca_current = emb_qca[:, :-1, :]  # Past Interaction History

        # 1. Question Model (LSTM)
        que_h = self.dropout_layer(self.que_lstm_layer(emb_qca_current)[0])
        que_outputs = self.get_outputs(
            emb_qc_shift, que_h, data, add_name="", model_type="question"
        )
        outputs = que_outputs

        # 2. Concept Model (LSTM)
        emb_ca = torch.cat(
            [
                emb_c.mul((1 - r).unsqueeze(-1).repeat(1, 1, self.emb_size)),
                emb_c.mul((r).unsqueeze(-1).repeat(1, 1, self.emb_size)),
            ],
            dim=-1,
        )
        emb_ca_current = emb_ca[:, :-1, :]

        concept_h = self.dropout_layer(self.concept_lstm_layer(emb_ca_current)[0])
        concept_outputs = self.get_outputs(
            emb_qc_shift, concept_h, data, add_name="", model_type="concept"
        )

        outputs["y_concept_all"] = concept_outputs["y_concept_all"]
        outputs["y_concept_next"] = concept_outputs["y_concept_next"]

        return outputs

    def get_attention_weights(self):
        return self.polya_emb.get_attention_weights()


# ------------------------------------------------------------------------------
# 6. PolyaQIKTV5 (Wrapper with Aux Loss)
# ------------------------------------------------------------------------------
class PolyaQIKTV5(QueBaseModel):
    def __init__(
        self,
        num_q,
        num_c,
        emb_size,
        dropout=0.1,
        emb_type="qaid",
        emb_path="",
        flag_load_emb=False,
        flag_emb_freezed=False,
        pretrain_dim=768,
        device="cpu",
        seed=0,
        mlp_layer_num=1,
        other_config={},
        **kwargs,
    ):

        model_name = "polya_qikt_v5"
        debug_print(
            f"Initializing PolyaQIKTV5 (State-Aware MoE + QIKT + Balancing)...",
            fuc_name=model_name,
        )

        super().__init__(
            model_name=model_name,
            emb_type=emb_type,
            emb_path=emb_path,
            pretrain_dim=pretrain_dim,
            device=device,
            seed=seed,
        )

        # Init config
        if "output_mode" not in other_config:
            other_config["output_mode"] = "an"

        # Determine Dimensions
        raw_emb_size = pretrain_dim if flag_load_emb else emb_size
        model_emb_size = emb_size

        self.model = PolyaQIKTNetV5(
            num_q=num_q,
            num_c=num_c,
            raw_emb_size=raw_emb_size,
            model_emb_size=model_emb_size,
            dropout=dropout,
            emb_type=emb_type,
            emb_path=emb_path,
            flag_load_emb=flag_load_emb,
            flag_emb_freezed=flag_emb_freezed,
            pretrain_dim=pretrain_dim,
            device=device,
            mlp_layer_num=mlp_layer_num,
            other_config=other_config,
        )
        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")
        self.eval_result = {}

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

        # Calculate Main Losses (QIKT has 4 losses combined)
        loss_q_all = self.get_loss(
            outputs["y_question_all"],
            data_new["rshft"],
            data_new["sm"],
            weighted_loss=weighted_loss,
        )
        loss_c_all = self.get_loss(
            outputs["y_concept_all"], data_new["rshft"], data_new["sm"]
        )
        loss_q_next = self.get_loss(
            outputs["y_question_next"],
            data_new["rshft"],
            data_new["sm"],
            weighted_loss=weighted_loss,
        )
        loss_c_next = self.get_loss(
            outputs["y_concept_next"], data_new["rshft"], data_new["sm"]
        )
        loss_kt = self.get_loss(
            outputs["y"], data_new["rshft"], data_new["sm"], weighted_loss=weighted_loss
        )

        def get_loss_lambda(x):
            return self.model.other_config.get(f"{x}", 0)

        loss_c_all_lambda = get_loss_lambda("loss_c_all_lambda")
        loss_c_next_lambda = get_loss_lambda("loss_c_next_lambda")
        loss_q_all_lambda = get_loss_lambda("loss_q_all_lambda")
        loss_q_next_lambda = get_loss_lambda("loss_q_next_lambda")

        if self.model.output_mode == "an_irt":
            main_loss = (
                loss_kt
                + loss_q_all_lambda * loss_q_all
                + loss_c_all_lambda * loss_c_all
                + loss_c_next_lambda * loss_c_next
            )
        else:
            main_loss = (
                loss_kt
                + loss_q_all_lambda * loss_q_all
                + loss_c_all_lambda * loss_c_all
                + loss_c_next_lambda * loss_c_next
                + loss_q_next_lambda * loss_q_next
            )

        # Add Aux Loss (Load Balancing)
        gate_probs = self.model.polya_emb.last_gate_probs
        if gate_probs is not None:
            aux_loss = self.get_load_balancing_loss(gate_probs)
            total_loss = main_loss + (0.01 * aux_loss)  # lambda = 0.01
        else:
            total_loss = main_loss

        return outputs["y"], total_loss

    def predict(self, dataset, batch_size, return_ts=False, process=True):
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            y_trues = []
            y_pred_dict = {}
            for data in test_loader:
                outputs, data_new = self.predict_one_step(
                    data, return_details=True, process=process
                )

                for key in outputs:
                    if not key.startswith("y"):
                        continue
                    elif key not in y_pred_dict:
                        y_pred_dict[key] = []
                    y = torch.masked_select(outputs[key], data_new["sm"]).detach().cpu()
                    y_pred_dict[key].append(y.numpy())

                t = (
                    torch.masked_select(data_new["rshft"], data_new["sm"])
                    .detach()
                    .cpu()
                )
                y_trues.append(t.numpy())

        results = y_pred_dict
        for key in results:
            results[key] = np.concatenate(results[key], axis=0)
        ts = np.concatenate(y_trues, axis=0)
        results["ts"] = ts
        return results

    def predict_one_step(
        self, data, return_details=False, process=True, return_raw=False
    ):
        data_new = self.batch_to_device(data, process=process)
        outputs = self.model(
            data_new["cq"].long(), data_new["cc"], data_new["cr"].long(), data=data_new
        )

        output_c_all_lambda = self.model.other_config.get("output_c_all_lambda", 1)
        output_c_next_lambda = self.model.other_config.get("output_c_next_lambda", 1)
        output_q_all_lambda = self.model.other_config.get("output_q_all_lambda", 1)
        output_q_next_lambda = self.model.other_config.get("output_q_next_lambda", 0)

        if self.model.output_mode == "an_irt":

            def sigmoid_inverse(x, epsilon=1e-8):
                return torch.log(x / (1 - x + epsilon) + epsilon)

            y = (
                sigmoid_inverse(outputs["y_question_all"]) * output_q_all_lambda
                + sigmoid_inverse(outputs["y_concept_all"]) * output_c_all_lambda
                + sigmoid_inverse(outputs["y_concept_next"]) * output_c_next_lambda
            )
            y = torch.sigmoid(y)
        else:
            y = (
                outputs["y_question_all"] * output_q_all_lambda
                + outputs["y_concept_all"] * output_c_all_lambda
                + outputs["y_concept_next"] * output_c_next_lambda
                + outputs["y_question_next"] * output_q_next_lambda
            )
            y = y / (
                output_q_all_lambda
                + output_c_all_lambda
                + output_c_next_lambda
                + output_q_next_lambda
            )

        outputs["y"] = y

        if return_details:
            return outputs, data_new
        else:
            return outputs["y"]

    def get_attention_weights(self):
        return self.model.get_attention_weights()
