import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


def _freeze_encoder_bn(model):
    """Force any frozen visual encoder into eval() so BatchNorm running stats
    do not drift during training. Handles both the CAVR pipeline (`.encoder`)
    and the baseline wrappers (`._encoder`, loaded lazily — may be None)."""
    for attr in ("encoder", "_encoder"):
        enc = getattr(model, attr, None)
        if isinstance(enc, nn.Module):
            enc.eval()


class BCTrainer:
    """Behavioral cloning trainer.

    Trains an MLP policy head on top of frozen visual encoders using MSE loss
    between predicted and expert actions.
    """

    def __init__(self, model, cfg, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg["training"]

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            trainable,
            lr=float(self.cfg["lr"]),
            weight_decay=float(self.cfg["weight_decay"]),
        )
        self.loss_fn = nn.MSELoss()
        self.logger = None

    def setup_wandb(self, project, run_name):
        try:
            import wandb
            self.logger = wandb.init(project=project, name=run_name, config=self.cfg)
        except ImportError:
            pass

    def train(self, dataset, task_description=None):
        val_size = max(1, int(0.1 * len(dataset)))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.cfg["seed"]),
        )

        default_workers = 0 if sys.platform == "darwin" else 4
        num_workers = int(self.cfg.get("num_workers", default_workers))
        pin_memory = self.device != "cpu"
        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=max(0, num_workers // 2),
            pin_memory=pin_memory,
        )

        ckpt_dir = self.cfg["checkpoint_dir"]
        os.makedirs(ckpt_dir, exist_ok=True)

        best_val_loss = float("inf")

        for epoch in range(self.cfg["num_epochs"]):
            train_loss = self._train_epoch(train_loader, task_description)

            log = {"epoch": epoch, "train_loss": train_loss}

            if (epoch + 1) % self.cfg["eval_freq"] == 0 or epoch == 0:
                val_loss = self._eval_epoch(val_loader, task_description)
                log["val_loss"] = val_loss

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(os.path.join(ckpt_dir, "best.pt"), epoch)

                print(
                    f"Epoch {epoch+1}/{self.cfg['num_epochs']} | "
                    f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{self.cfg['num_epochs']} | "
                    f"train_loss={train_loss:.6f}"
                )

            if self.logger:
                import wandb
                wandb.log(log)

        self._save_checkpoint(os.path.join(ckpt_dir, "final.pt"), epoch)
        return best_val_loss

    def _train_epoch(self, loader, task_description):
        self.model.train()
        _freeze_encoder_bn(self.model)
        total_loss = 0.0
        n = 0

        for images, proprios, actions in loader:
            images = images.to(self.device)
            proprios = proprios.to(self.device)
            actions = actions.to(self.device)

            pred_actions = self.model(images, proprios, task_description)
            loss = self.loss_fn(pred_actions, actions)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * images.shape[0]
            n += images.shape[0]

        return total_loss / n

    @torch.no_grad()
    def _eval_epoch(self, loader, task_description):
        self.model.eval()
        total_loss = 0.0
        n = 0

        for images, proprios, actions in loader:
            images = images.to(self.device)
            proprios = proprios.to(self.device)
            actions = actions.to(self.device)

            pred_actions = self.model(images, proprios, task_description)
            loss = self.loss_fn(pred_actions, actions)

            total_loss += loss.item() * images.shape[0]
            n += images.shape[0]

        return total_loss / n

    def _save_checkpoint(self, path, epoch):
        frozen_prefixes = (
            "encoder.model",
            "_encoder",
            "masker",
            "_grounding_model",
            "_sam_predictor",
        )
        trainable_state = {
            k: v for k, v in self.model.state_dict().items()
            if not any(p in k for p in frozen_prefixes)
        }
        torch.save({
            "epoch": epoch,
            "model_state_dict": trainable_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"]
