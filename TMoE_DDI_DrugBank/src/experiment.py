import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import load_experiment_data
from .evaluation import evaluate
from .model import TMoEDDI
from .utils import set_random_seed


def build_model(args):
    model = TMoEDDI(
        feature_dim=args.feature_dim,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
        num_relations=args.num_relations,
        molecular_features=args.molecular_features,
    )
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


def checkpoint_path(args, stage):
    return Path(args.root_dir) / "checkpoints" / f"{stage}_best_fold_{args.fold}.pth"


def run_experiment(args):
    set_random_seed(args.seed)
    graph, train_loader, valid_loader, test_loader = load_experiment_data(args)
    model, optimizer = build_model(args)

    if args.cuda:
        model.cuda()
        graph = graph.to("cuda")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    stage1_path = checkpoint_path(args, "stage1")
    stage2_path = checkpoint_path(args, "stage2")
    skip_stage1 = False
    skip_stage2 = False

    if stage2_path.exists() and not args.force_train:
        print(f"Loading Stage 2 checkpoint: {stage2_path}")
        model.init_gate_layer()
        checkpoint = torch.load(stage2_path, map_location="cuda" if args.cuda else "cpu")
        model.load_state_dict(remap_checkpoint_keys(checkpoint))
        skip_stage1 = True
        skip_stage2 = True
    elif stage1_path.exists() and not args.force_train:
        print(f"Loading Stage 1 checkpoint: {stage1_path}")
        checkpoint = torch.load(stage1_path, map_location="cuda" if args.cuda else "cpu")
        model.load_state_dict(remap_checkpoint_keys(checkpoint))
        skip_stage1 = True

    if not skip_stage1:
        print("Training Stage 1 experts...")
        best_acc = 0.0
        patience = 0
        best_state = copy.deepcopy(model.state_dict())
        for epoch in range(args.epochs):
            model.train()
            for batch in train_loader:
                labels = torch.as_tensor(batch[2], dtype=torch.long)
                if args.cuda:
                    labels = labels.cuda()
                optimizer.zero_grad()
                _, e1, e2, e3, _, _ = model(graph, batch, stage=1)
                ce_loss = loss_fn(e1, labels) + loss_fn(e2, labels) + loss_fn(e3, labels)
                p1, p2, p3 = F.softmax(e1, dim=1).detach(), F.softmax(e2, dim=1).detach(), F.softmax(e3, dim=1).detach()
                lp1, lp2, lp3 = F.log_softmax(e1, dim=1), F.log_softmax(e2, dim=1), F.log_softmax(e3, dim=1)
                mutual_loss = (
                    kl_loss(lp1, p2) + kl_loss(lp1, p3)
                    + kl_loss(lp2, p1) + kl_loss(lp2, p3)
                    + kl_loss(lp3, p1) + kl_loss(lp3, p2)
                ) / 3.0
                loss = ce_loss + args.beta * mutual_loss
                loss.backward()
                optimizer.step()
            valid_metrics, _, _, _ = evaluate(model, valid_loader, graph, args, stage=1)
            acc = valid_metrics["All"]["ACC"]
            print(f"[Stage 1] epoch={epoch + 1:03d} val_acc={acc:.4f}")
            if acc > best_acc:
                best_state = copy.deepcopy(model.state_dict())
                best_acc = acc
                patience = 0
            else:
                patience += 1
            if patience >= args.convergence_patience:
                break
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), stage1_path)
    else:
        print("Stage 1 training skipped.")

    print("Evaluating Stage 1...")
    s1_fused, s1_e1, s1_e2, s1_e3 = evaluate(model, test_loader, graph, args, stage=1)

    if not skip_stage2:
        print("Training Stage 2 gate...")
        if not hasattr(model, "gate_layer"):
            model.init_gate_layer()
        for name, param in model.named_parameters():
            param.requires_grad = "gate_layer" in name
        gate_optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.gate_lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(gate_optimizer, T_max=50, eta_min=1e-6)
        best_acc = 0.0
        patience = 0
        best_state = copy.deepcopy(model.state_dict())
        for epoch in range(args.stage2_epochs):
            model.train()
            for batch in train_loader:
                labels = torch.as_tensor(batch[2], dtype=torch.long)
                if args.cuda:
                    labels = labels.cuda()
                gate_optimizer.zero_grad()
                logits, _, _, _, _, _ = model(graph, batch, stage=2)
                loss = loss_fn(logits, labels)
                loss.backward()
                gate_optimizer.step()
            scheduler.step()
            valid_metrics, _, _, _ = evaluate(model, valid_loader, graph, args, stage=2)
            acc = valid_metrics["All"]["ACC"]
            print(f"[Stage 2] epoch={epoch + 1:03d} val_acc={acc:.4f}")
            if acc > best_acc:
                best_state = copy.deepcopy(model.state_dict())
                best_acc = acc
                patience = 0
                torch.save(model.state_dict(), stage2_path)
            else:
                patience += 1
            if patience >= 25:
                break
        model.load_state_dict(best_state)
    else:
        print("Stage 2 training skipped.")

    print("Evaluating Stage 2...")
    s2_fused, s2_e1, s2_e2, s2_e3 = evaluate(model, test_loader, graph, args, stage=2)
    result = {
        "seed": str(args.fold),
        "Stage1": {"Expert1": s1_e1, "Expert2": s1_e2, "Expert3": s1_e3, "Fused": s1_fused},
        "Stage2": {"Expert1": s2_e1, "Expert2": s2_e2, "Expert3": s2_e3, "Fused": s2_fused},
    }
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def remap_checkpoint_keys(state_dict):
    return {key.replace(".model.", ".layers."): value for key, value in state_dict.items()}
