#!/usr/bin/env python3
import torch
from pathlib import Path
import timm

# -----------------------
# Paths
# -----------------------
MAIN_DIR = Path("/home/dexter-morgan/PROJECTS/FL_Encrypt/simulations/model")
MAIN_DIR.mkdir(parents=True, exist_ok=True)
MAIN_SAVE_PATH = MAIN_DIR / "convnextv2_base_fcmae384.pt"

GLOBAL_DIR = Path("/home/dexter-morgan/PROJECTS/FL_Encrypt/simulations/master/initial_global")
GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
GLOBAL_SAVE_PATH = GLOBAL_DIR / "initial_global.pt"

# -----------------------
# Load ConvNeXt-V2 Base FCMAE pretrained on IN22k → IN1k
# -----------------------
model_name = "convnextv2_base.fcmae_ft_in22k_in1k_384"
model = timm.create_model(model_name, pretrained=True)
model.eval()  # set model to evaluation mode

# -----------------------
# Save the state_dict in both locations
# -----------------------
torch.save(model.state_dict(), MAIN_SAVE_PATH)
torch.save(model.state_dict(), GLOBAL_SAVE_PATH)

print(f"Saved ConvNeXt-V2 Base FCMAE state_dict to:\n- {MAIN_SAVE_PATH}\n- {GLOBAL_SAVE_PATH}")