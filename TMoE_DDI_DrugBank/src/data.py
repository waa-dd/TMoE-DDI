from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from .utils import normalize_rows


class InteractionDataset(Dataset):
    def __init__(self, triples):
        self.drug_a = triples[:, 0]
        self.drug_b = triples[:, 1]
        self.label = triples[:, 2]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.drug_a[index], self.drug_b[index], self.label[index]


def read_drug_ids(data_dir: Path):
    frame = pd.read_csv(data_dir / "drug_list.csv")
    return frame["drug_id"].astype(str).tolist()


def load_split(data_dir: Path, fold: int, split: str, drug_to_idx):
    frame = pd.read_csv(data_dir / f"fold_{fold}" / f"{split}.csv")
    triples = np.empty((len(frame), 3), dtype=np.int64)
    triples[:, 0] = [drug_to_idx[drug] for drug in frame["d1"]]
    triples[:, 1] = [drug_to_idx[drug] for drug in frame["d2"]]
    triples[:, 2] = frame["type"].astype(np.int64).to_numpy()
    np.random.shuffle(triples)
    return triples


def load_molecular_embeddings(root: Path, fold: int, drug_ids):
    embeddings = np.load(root / "molecular_embeddings" / f"trimnet_embedding_fold_{fold}.npy")
    embedding_ids = np.load(root / "molecular_embeddings" / "drug_ids.npy").tolist()
    aligned = np.array([embeddings[embedding_ids.index(drug_id)] for drug_id in drug_ids])
    return aligned


def build_interaction_graph(features, train_triples):
    normalized = normalize_rows(features)
    x = torch.tensor(normalized, dtype=torch.float)

    edge_index = []
    edge_type = []
    for drug_a, drug_b, relation in train_triples:
        edge_index.append([int(drug_a), int(drug_b)])
        edge_type.append(int(relation))
        edge_index.append([int(drug_b), int(drug_a)])
        edge_type.append(int(relation))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type)


def load_experiment_data(args):
    root = Path(args.root_dir)
    data_dir = root / "data"
    drug_ids = read_drug_ids(data_dir)
    drug_to_idx = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}

    train_triples = load_split(data_dir, args.fold, "train", drug_to_idx)
    valid_triples = load_split(data_dir, args.fold, "valid", drug_to_idx)
    test_triples = load_split(data_dir, args.fold, "test", drug_to_idx)

    args.num_relations = 86
    type_counts = Counter(train_triples[:, 2].tolist())
    args.head_relations = {rel for rel in range(args.num_relations) if type_counts.get(rel, 0) > 50}
    args.middle_relations = {rel for rel in range(args.num_relations) if 15 <= type_counts.get(rel, 0) <= 50}
    args.tail_relations = {rel for rel in range(args.num_relations) if type_counts.get(rel, 0) < 15}

    loader_args = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "drop_last": False,
    }
    train_loader = DataLoader(InteractionDataset(train_triples), **loader_args)
    valid_loader = DataLoader(InteractionDataset(valid_triples), **loader_args)
    test_loader = DataLoader(InteractionDataset(test_triples), **loader_args)

    molecular_features = load_molecular_embeddings(root, args.fold, drug_ids)
    graph = build_interaction_graph(molecular_features, train_triples)
    args.feature_dim = graph.x.shape[1]
    args.num_drugs = graph.x.shape[0]
    args.molecular_features = torch.tensor(molecular_features, dtype=torch.float)
    return graph, train_loader, valid_loader, test_loader
