import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import timm
from torch.cuda.amp import GradScaler, autocast
import wandb


# ┌─────────────────────────────────────────────────────────────
# CONFIG
# └─────────────────────────────────────────────────────────────

ROOT_PATH = "/home/dexter-morgan/PROJECTS/FL_Encrypt/simulations"
MODEL_PATH = os.path.join(ROOT_PATH, "model", "convnextv2_base_fcmae384.pt")

WORKER1_DATA = os.path.join(ROOT_PATH, "workers", "data", "worker_1", "data")
WORKER2_DATA = os.path.join(ROOT_PATH, "workers", "data", "worker_2", "data")

NUM_CLASSES = 4        # Cyst, Normal, Stone, Tumor
IMG_SIZE = 384         # matches FCMAE‑384
CROP_SIZE = 192        # RandomResizedCrop target
BATCH_SIZE = 8
NUM_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = True

# wandb
USE_WANDB = True
WANDB_PROJECT = "FL_Encrypt_finetune"
WANDB_NAME = "convnextv2_base_fcmae384_kidney"


# ┌─────────────────────────────────────────────────────────────
# Custom normalization (per worker)
# └─────────────────────────────────────────────────────────────

transforms_worker1 = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.RandomResizedCrop(CROP_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.1283, 0.1283, 0.1283],
        std=[0.2063, 0.2063, 0.2063],
    ),
])

transforms_worker2 = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.RandomResizedCrop(CROP_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.1433, 0.1433, 0.1433],
        std=[0.2244, 0.2244, 0.2244],
    ),
])


def load_data(data_dir, transforms):
    """
    Use torchvision.datasets.ImageFolder with your custom transforms.
    Assumes data_dir has subfolders: Cyst, Normal, Stone, Tumor.
    """
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transforms,
    )
    return dataset


def load_data_split(data_dir, transforms, split_ratio=0.8, seed=42):
    dataset = load_data(data_dir, transforms)
    total = len(dataset)
    train_len = int(total * split_ratio)
    val_len = total - train_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(seed),
    )
    return train_set, val_set


# ┌─────────────────────────────────────────────────────────────
# Model: ConvNeXtV2 base FCMAE‑384 from .pt
def load_convnextv2_fcmae384(model_path, num_classes):

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    if "model" in state_dict:
        state_dict = state_dict["model"]

    backbone = timm.create_model(
        "convnextv2_base.fcmae_ft_in22k_in1k",
        pretrained=False,
        num_classes=0,
    )

    backbone.load_state_dict(state_dict, strict=False)

    class FixedHeadModel(nn.Module):
        def __init__(self, backbone, num_classes):
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Linear(backbone.num_features, num_classes)

        def forward(self, x):
            x = self.backbone.forward_features(x)
            x = self.backbone.forward_head(x, pre_logits=True)
            return self.classifier(x)

    model = FixedHeadModel(backbone, num_classes)
    print(f"✅ Fixed model: {backbone.num_features} -> {num_classes}")

    return model



# ┌─────────────────────────────────────────────────────────────
# Train / Val loop
# └─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_seen += x.size(0)

    return total_loss / total_seen, total_correct / total_seen


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total_seen += x.size(0)

    return total_loss / total_seen, total_correct / total_seen


# ┌─────────────────────────────────────────────────────────────
# Main
# └─────────────────────────────────────────────────────────────

def main():
    print("Building datasets with ImageFolder and custom transforms...")

    # proper data split for both workers
    train_w1, val_w1 = load_data_split(WORKER1_DATA, transforms_worker1)
    train_w2, val_w2 = load_data_split(WORKER2_DATA, transforms_worker2)

    full_train = ConcatDataset([train_w1, train_w2])
    full_val = ConcatDataset([val_w1, val_w2])  # Combined validation

    train_loader = DataLoader(
        full_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        full_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Train samples: {len(full_train)} | Val samples: {len(full_val)}")

    # model
    print("Loading ConvNeXtV2 FCMAE‑384 from checkpoint...")
    model = load_convnextv2_fcmae384(MODEL_PATH, NUM_CLASSES).to(DEVICE)

    # loss + optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler() if AMP else None

    # wandb (optional)
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_NAME,
            config={
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "img_size": IMG_SIZE,
                "crop_size": CROP_SIZE,
                "amp": AMP,
            }
        )

    # training loop
    best_val_acc = 0.0
    no_improve = 0

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, DEVICE, AMP
        )

        # use val_loader only if you want separate worker‑2 validation
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        scheduler.step()  # adjust LR after validation

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | Acc: {val_acc:.4f}"
        )

        if USE_WANDB:
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(
                model.state_dict(),
                os.path.join(ROOT_PATH, "model", "convnextv2_base_fcmae384.pt"),
            )
            print(f"New best val accuracy: {best_val_acc:.4f} | checkpoint saved.")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training finished. Final best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
