import torch
import timm
import torch.nn as nn
from pathlib import Path

NUM_CLASSES = 4

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR.parent / "model" / "convnextv2_base_fcmae384.pt"

def get_model(device="cuda"):

    # create architecture
    model = timm.create_model(
        "convnextv2_base.fcmae_ft_in22k_in1k_384",
        pretrained=False
    )
    # get input size of classifier
    in_features = model.head.fc.in_features
    # replace classifier head
    model.head.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.Dropout(0.4),
        nn.Linear(512, NUM_CLASSES)
    )
    # load pretrained checkpoint
    if Path(WEIGHTS_PATH).exists():
        checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        new_state = {}
        for k, v in checkpoint.items():
            if k.startswith("module."):
                k = k.replace("module.", "")
            new_state[k] = v
        model.load_state_dict(new_state, strict=False)
        print(f"Loaded pretrained weights from {WEIGHTS_PATH}")
    else:
        print("WARNING: pretrained weights not found")
    model = model.to(device)
    return model

def get_parameters(model):
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
