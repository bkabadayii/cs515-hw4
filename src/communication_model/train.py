import argparse
import random
import numpy as np

import torch
import torch.nn.functional as F

from communication_model.model import NeuralCommunicationSystem


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_messages(
    batch_size: int,
    message_len: int,
    num_symbols: int,
    device: torch.device,
):
    """
    Randomly sample messages m in {0,...,7}^{B x 4}.
    """
    return torch.randint(
        low=0,
        high=num_symbols,
        size=(batch_size, message_len),
        device=device,
    )


@torch.no_grad()
def evaluate(
    model,
    device,
    batch_size: int = 1024,
    num_batches: int = 50,
    message_len: int = 4,
    num_symbols: int = 8,
):
    model.eval()

    total_loss = 0.0
    total_token_acc = 0.0
    total_message_acc = 0.0

    for _ in range(num_batches):
        m = sample_messages(
            batch_size=batch_size,
            message_len=message_len,
            num_symbols=num_symbols,
            device=device,
        )

        logits = model(m)

        loss = F.cross_entropy(
            logits.reshape(-1, num_symbols),
            m.reshape(-1),
        )

        pred = logits.argmax(dim=-1)

        token_acc = (pred == m).float().mean()
        message_acc = (pred == m).all(dim=1).float().mean()

        total_loss += loss.item()
        total_token_acc += token_acc.item()
        total_message_acc += message_acc.item()

    total_loss /= num_batches
    total_token_acc /= num_batches
    total_message_acc /= num_batches

    return total_loss, total_token_acc, total_message_acc


def train(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"Using device: {device}")

    model = NeuralCommunicationSystem(
        num_symbols=args.num_symbols,
        message_len=args.message_len,
        num_rounds=args.num_rounds,
        sigma=args.sigma,
        d_symbol=args.d_symbol,
        d_round=args.d_round,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_message_acc = 0.0

    for step in range(1, args.steps + 1):
        model.train()

        m = sample_messages(
            batch_size=args.batch_size,
            message_len=args.message_len,
            num_symbols=args.num_symbols,
            device=device,
        )

        logits = model(m)

        loss = F.cross_entropy(
            logits.reshape(-1, args.num_symbols),
            m.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                token_acc = (pred == m).float().mean()
                message_acc = (pred == m).all(dim=1).float().mean()

            print(
                f"step {step:05d} | "
                f"train loss {loss.item():.4f} | "
                f"token acc {token_acc.item():.4f} | "
                f"message acc {message_acc.item():.4f}"
            )

        if step % args.eval_every == 0:
            val_loss, val_token_acc, val_message_acc = evaluate(
                model=model,
                device=device,
                batch_size=args.eval_batch_size,
                num_batches=args.eval_batches,
                message_len=args.message_len,
                num_symbols=args.num_symbols,
            )

            print(
                f"[eval] step {step:05d} | "
                f"loss {val_loss:.4f} | "
                f"token acc {val_token_acc:.4f} | "
                f"message acc {val_message_acc:.4f}"
            )

            if val_message_acc > best_message_acc:
                best_message_acc = val_message_acc

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "best_message_acc": best_message_acc,
                    },
                    args.save_path,
                )

                print(
                    f"Saved best model to {args.save_path} "
                    f"with message acc {best_message_acc:.4f}"
                )

    print("Training finished.")
    print(f"Best validation message accuracy: {best_message_acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()

    # Problem setup
    parser.add_argument("--num_symbols", type=int, default=8)
    parser.add_argument("--message_len", type=int, default=4)
    parser.add_argument("--num_rounds", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.5)

    # Model
    parser.add_argument("--d_symbol", type=int, default=16)
    parser.add_argument("--d_round", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Logging
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default="best_comm_model.pt")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
