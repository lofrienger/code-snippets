#!/usr/bin/env python3
"""Train a small convolutional network on MNIST."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = F.relu(F.max_pool2d(self.conv1(inputs), 2))
        inputs = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(inputs)), 2))
        inputs = torch.flatten(inputs, 1)
        inputs = F.relu(self.fc1(inputs))
        inputs = F.dropout(inputs, training=self.training)
        return F.log_softmax(self.fc2(inputs), dim=1)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--epochs", type=positive_int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=positive_int, default=10)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        help="optional output path for the trained state dictionary",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run one training batch",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def train(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    log_interval: int,
    dry_run: bool,
) -> None:
    model.train()
    for batch_index, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_index % log_interval == 0:
            processed = batch_index * len(data)
            percent = 100.0 * batch_index / len(loader)
            print(
                f"Train epoch {epoch}: {processed}/{len(loader.dataset)} "
                f"({percent:.0f}%) loss={loss.item():.6f}"
            )
        if dry_run:
            break


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        args.data_dir, train=True, download=True, transform=transform
    )
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_options,
    )
    model = Net().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum
    )
    for epoch in range(1, args.epochs + 1):
        train(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            args.log_interval,
            args.dry_run,
        )
        if args.dry_run:
            break

    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print(f"Saved model to {args.save_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
