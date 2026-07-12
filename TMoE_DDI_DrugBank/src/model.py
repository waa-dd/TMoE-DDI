import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import RGCNConv


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        return F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))


class ExpertClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_ratio=1.0):
        super().__init__()
        hidden_channels = int(in_channels * hidden_ratio)
        self.layers = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ELU(),
            nn.Dropout(p=0.1),
            NormedLinear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class TMoEDDI(nn.Module):
    def __init__(self, feature_dim, hidden1, hidden2, dropout, num_relations, molecular_features):
        super().__init__()
        self.num_relations = num_relations
        self.dropout = dropout
        self.logit_scale = 30.0
        self.hidden2 = hidden2
        self.molecular_features = molecular_features.float()

        self.shared_encoder_1 = RGCNConv(feature_dim, hidden1, num_relations=num_relations)
        self.shared_encoder_2 = RGCNConv(hidden1, hidden2, num_relations=num_relations)
        self.res_proj = nn.Linear(hidden1, hidden2)

        self.expert1_gnn = RGCNConv(hidden2, hidden2, num_relations=num_relations)
        self.expert2_gnn = RGCNConv(hidden2, hidden2, num_relations=num_relations)
        self.expert3_gnn = RGCNConv(hidden2, hidden2, num_relations=num_relations)

        self.align_x1 = nn.Linear(hidden1, hidden2)
        self.align_x2 = nn.Linear(hidden2, hidden2)
        self.align_x3 = nn.Linear(hidden2, hidden2)

        expert_input_dim = (hidden2 + self.molecular_features.shape[1]) * 2 + (2 * hidden2)
        self.classifier1 = ExpertClassifier(expert_input_dim, num_relations, hidden_ratio=1.0)
        self.classifier2 = ExpertClassifier(expert_input_dim, num_relations, hidden_ratio=1.0)
        self.classifier3 = ExpertClassifier(expert_input_dim, num_relations, hidden_ratio=1.0)

    def init_gate_layer(self):
        gate_input_dim = (self.hidden2 + self.molecular_features.shape[1]) * 2 + (2 * self.hidden2)
        self.gate_layer = nn.Sequential(
            nn.Linear(gate_input_dim, self.hidden2),
            nn.LayerNorm(self.hidden2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden2, 3),
        )
        nn.init.zeros_(self.gate_layer[-1].weight)
        nn.init.zeros_(self.gate_layer[-1].bias)
        self.gate_layer.to(next(self.parameters()).device)

    @staticmethod
    def _interaction_features(graph_head, graph_tail, mol_head, mol_tail):
        base = torch.cat((graph_head, mol_head, graph_tail, mol_tail), dim=1)
        return torch.cat((base, torch.abs(graph_head - graph_tail), graph_head * graph_tail), dim=1)

    def forward(self, graph, batch, stage=1):
        self.molecular_features = self.molecular_features.to(graph.x.device)
        x, edge_index, edge_type = graph.x, graph.edge_index, graph.edge_type.to(graph.x.device)

        x1 = F.relu(self.shared_encoder_1(x, edge_index, edge_type))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2_raw = self.shared_encoder_2(x1, edge_index, edge_type)
        x2 = F.relu(x2_raw + self.res_proj(x1))
        x2 = F.dropout(x2, self.dropout, training=self.training)

        expert1 = torch.tanh(self.align_x1(x1)) * self.expert1_gnn(x2, edge_index, edge_type)
        expert2 = torch.tanh(self.align_x2(x2)) * self.expert2_gnn(x2, edge_index, edge_type)
        expert3 = torch.tanh(self.align_x3(x2)) * self.expert3_gnn(x2, edge_index, edge_type)

        drug_a = torch.as_tensor(batch[0], dtype=torch.long, device=x.device)
        drug_b = torch.as_tensor(batch[1], dtype=torch.long, device=x.device)
        mol_a = self.molecular_features[drug_a]
        mol_b = self.molecular_features[drug_b]

        e1_input = self._interaction_features(expert1[drug_a], expert1[drug_b], mol_a, mol_b)
        e2_input = self._interaction_features(expert2[drug_a], expert2[drug_b], mol_a, mol_b)
        e3_input = self._interaction_features(expert3[drug_a], expert3[drug_b], mol_a, mol_b)

        logits_e1 = self.logit_scale * self.classifier1(e1_input)
        logits_e2 = self.logit_scale * self.classifier2(e2_input)
        logits_e3 = self.logit_scale * self.classifier3(e3_input)

        gate_weights = None
        gate_input = None
        if stage == 1:
            fused_logits = (logits_e1 + logits_e2 + logits_e3) / 3.0
        else:
            if not hasattr(self, "gate_layer"):
                raise RuntimeError("Gate layer is not initialized. Call init_gate_layer() before Stage 2.")
            gate_input = self._interaction_features(x2[drug_a], x2[drug_b], mol_a, mol_b)
            gate_weights = F.softmax(self.gate_layer(gate_input), dim=1)
            fused_logits = (
                gate_weights[:, 0:1] * logits_e1
                + gate_weights[:, 1:2] * logits_e2
                + gate_weights[:, 2:3] * logits_e3
            )
        return fused_logits, logits_e1, logits_e2, logits_e3, gate_weights, gate_input
