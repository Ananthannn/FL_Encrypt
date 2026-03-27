import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from model import get_model
from opacus.utils.batch_memory_manager import BatchMemoryManager

BATCH_SIZE = 2
EPOCHS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from torch.utils.data import DataLoader, Subset
import random

def load_data(data_dir):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((384,384)),
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

    # ---- pick 10 images per class ----
    class_indices = {i: [] for i in range(len(dataset.classes))}

    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    selected_indices = []
    for cls in class_indices:
        selected_indices += random.sample(class_indices[cls], 10)

    subset = Subset(dataset, selected_indices)

    loader = DataLoader(
        subset,
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
    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=1.1,
        max_grad_norm=0.5,
        poisson_sampling=False,
    )

    # cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
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