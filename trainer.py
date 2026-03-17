import os
import math
import time
import torch
from typing import Optional
from utils import save_checkpoint, load_checkpoint


class Trainer:
    def __init__(self, model: torch.nn.Module, args, device: torch.device, out_dir: Optional[str] = None):
        self.model = model
        self.args = args
        self.device = device
        self.out_dir = out_dir or "./sessions"
        os.makedirs(self.out_dir, exist_ok=True)

        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        self.start_epoch = 0
        self.best_val = math.inf

        if getattr(args, "resume", ""):
            if os.path.exists(args.resume):
                ckpt = load_checkpoint(args.resume, model=self.model, optimizer=self.optimizer, map_location=device)
                self.start_epoch = ckpt.get("epoch", 0)
                self.best_val = ckpt.get("best_val", math.inf)

    def _to_device(self, images, targets):
        images = [img.to(self.device) for img in images]
        for t in targets:
            for k, v in t.items():
                if isinstance(v, torch.Tensor):
                    t[k] = v.to(self.device)
        return images, targets

    def train_one_epoch(self, data_loader, epoch: int):
        self.model.train()
        running_loss = 0.0
        iters = 0
        start = time.time()
        for images, targets in data_loader:
            images, targets = self._to_device(images, targets)
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            running_loss += float(losses.item())
            iters += 1

        avg_loss = running_loss / max(1, iters)
        elapsed = time.time() - start
        print(f"Epoch {epoch} Train loss: {avg_loss:.4f} ({iters} iters, {elapsed:.1f}s)")
        return avg_loss

    def validate(self, data_loader):
        self.model.eval()
        running_loss = 0.0
        iters = 0
        with torch.no_grad():
            for images, targets in data_loader:
                images, targets = self._to_device(images, targets)
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                running_loss += float(losses.item())
                iters += 1

        avg_loss = running_loss / max(1, iters)
        print(f"Validation loss: {avg_loss:.4f} ({iters} iters)")
        return avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        path_last = os.path.join(self.out_dir, "checkpoint_last.pth")
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val": self.best_val,
            "args": vars(self.args) if self.args is not None else {},
        }
        save_checkpoint(state, path_last)
        if is_best:
            path_best = os.path.join(self.out_dir, "checkpoint_best.pth")
            save_checkpoint(state, path_best)

    def train(self, train_loader, val_loader=None):
        num_epochs = getattr(self.args, "epochs", 10)
        for epoch in range(self.start_epoch, num_epochs):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)

            self.scheduler.step()

            improved = False
            if val_loss is not None:
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    improved = True

            self.save_checkpoint(epoch, is_best=improved)

        print("Training complete")
