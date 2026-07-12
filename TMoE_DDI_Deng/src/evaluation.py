import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize


def grouped_metrics(y_true, logits, args):
    logits = np.array(logits).reshape((-1, args.num_relations))
    y_pred = np.argmax(logits, axis=1)
    y_true = np.array(y_true)

    metrics = {
        "All": {
            "ACC": float(accuracy_score(y_true, y_pred)),
            "F1": float(f1_score(y_true, y_pred, average="macro")),
            "Pre": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "Rec": float(recall_score(y_true, y_pred, average="macro")),
        }
    }

    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(args.num_relations)),
        zero_division=0,
    )

    def subset_metrics(relations):
        if not relations:
            return {"ACC": 0.0, "F1": 0.0, "Pre": 0.0, "Rec": 0.0}
        relations = list(relations)
        indices = [idx for idx, label in enumerate(y_true) if label in relations]
        return {
            "ACC": float(accuracy_score(y_true[indices], y_pred[indices])) if indices else 0.0,
            "F1": float(np.mean([f1s[rel] for rel in relations])),
            "Pre": float(np.mean([precisions[rel] for rel in relations])),
            "Rec": float(np.mean([recalls[rel] for rel in relations])),
        }

    metrics["Common"] = subset_metrics(args.head_relations)
    metrics["Fewer"] = subset_metrics(args.middle_relations)
    metrics["Rare"] = subset_metrics(args.tail_relations)

    try:
        y_true_bin = label_binarize(y_true, classes=list(range(args.num_relations)))
        probabilities = F.softmax(torch.tensor(logits), dim=1).numpy()
        metrics["aupr"] = float(average_precision_score(y_true_bin, probabilities, average="macro"))
    except Exception:
        metrics["aupr"] = 0.0

    return metrics


def evaluate(model, loader, graph, args, stage):
    model.eval()
    labels = []
    fused_logits = []
    expert1_logits = []
    expert2_logits = []
    expert3_logits = []

    with torch.no_grad():
        for batch in loader:
            label = torch.as_tensor(batch[2], dtype=torch.long)
            if args.cuda:
                label = label.cuda()

            fused, e1, e2, e3, _, _ = model(graph, batch, stage=stage)
            labels.extend(label.cpu().numpy().flatten().tolist())
            fused_logits.extend(fused.cpu().numpy().flatten().tolist())
            expert1_logits.extend(e1.cpu().numpy().flatten().tolist())
            expert2_logits.extend(e2.cpu().numpy().flatten().tolist())
            expert3_logits.extend(e3.cpu().numpy().flatten().tolist())

    return (
        grouped_metrics(labels, fused_logits, args),
        grouped_metrics(labels, expert1_logits, args),
        grouped_metrics(labels, expert2_logits, args),
        grouped_metrics(labels, expert3_logits, args),
    )
