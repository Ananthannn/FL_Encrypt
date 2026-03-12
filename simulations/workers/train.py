import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from model import get_model


BATCH_SIZE = 8
EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(data_dir):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(192, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.1283, 0.1283, 0.1283],
            std=[0.2063, 0.2063, 0.2063]
        )
    ])
# for worker 2 Mean: tensor([0.1433, 0.1433, 0.1433])
#Std: tensor([0.2244, 0.2244, 0.2244])
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print("Classes:", dataset.classes)
    print("Total samples:", len(dataset))
    return loader

def train(model, loader, device):
    # class weights (dataset imbalance handling)
    class_weights = torch.tensor([1.4,0.8,1.6,1.8]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # discriminative learning rates
    optimizer = optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": 1e-3},
            {"params": model.stages.parameters(), "lr": 5e-5},
        ],
        weight_decay=1e-4
    )
    
    # ------------------------
    # Differential Privacy Engine
    # ------------------------
    privacy_engine = PrivacyEngine(
        model,
        batch_size=loader.batch_size,
        sample_size=len(loader.dataset),
        alphas=[10, 100],
        noise_multiplier=1.1,  # adjust for target epsilon
        max_grad_norm=1.0,
    )
    privacy_engine.attach(optimizer)

    # cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )
    # mixed precision scaler
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # mixed precision forward
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            # accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            print(
                f"Epoch {epoch+1}/{EPOCHS} "
                f"Batch {batch_idx+1}/{len(loader)} "
                f"Loss {loss.item():.4f}"
            )
        scheduler.step()
        epoch_loss = running_loss / len(loader)
        epoch_acc = correct / total
        print(
            f"\nEpoch {epoch+1} Summary"
            f"\nLoss: {epoch_loss:.4f}"
            f"\nAccuracy: {epoch_acc:.4f}\n"
        )
    return model

if __name__ == "__main__":
    DATA_PATH = "data/worker_1/data"
    print("Loading model...")
    model = get_model(DEVICE)
    model.set_grad_checkpointing(True)
    print("Loading dataset...")
    trainloader = load_data(DATA_PATH)
    print("Training...")
    model = train(model, trainloader, DEVICE)
    print("Training complete.")