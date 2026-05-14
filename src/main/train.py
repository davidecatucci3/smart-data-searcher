import torch.nn.functional as F
import wandb
import torch
import time
import os

from transformers import get_cosine_schedule_with_warmup
from import_data import train_loader, test_loader
from hyperparameters import hyperparameters
from torch.amp import autocast, GradScaler
from model import Model

# prevents memory fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# set TF32 instead of FP32 (faster GPU matmul using TF32)
torch.set_float32_matmul_precision('high')

# set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# hyperparameters
lr = hyperparameters['lr']
batch_size = hyperparameters['batch_size']
d_i = hyperparameters['d_i']
d_t = hyperparameters['d_t']
d_e = hyperparameters['d_e']
alpha = hyperparameters['alpha']
epochs = hyperparameters['epochs']
max_test_batches = hyperparameters['max_test_batches']
top_k = hyperparameters['top_k']
accumulation_steps = hyperparameters['accumulation_steps']

# initialize WandB (used for saving data during training and see them live on the WandB platform)
wandb.init(
    entity="davidecatucci3-sapienza-universit-di-roma",
    project="smart data searcher",
    config={
        "learning_rate": lr,
        "batch_size": batch_size,
        "d_i": d_i,
        "d_t": d_t,
        "d_e": d_e,
        "alpha": alpha,
        "epochs": epochs,
        "max test batches": max_test_batches,
        "alpha": alpha,
        "top_k": top_k,
        "accumulation_steps": accumulation_steps,
        "dataset": "-",
        "architecture": "CLIP"
    }
)

wandb.define_metric("global_step") # define x-axis metric

# set model
model = Model().to(device) # move parameters to device (GPU)
scaler = GradScaler('cuda') # prevent the usage of autocast float16 to underflow by scaling gradient

# froze image encoder parameters (don't track gradient so won't be update it) -> Locked-image Tuning (LiT) technique
for param in model.img_encoder.parameters():
    param.requires_grad = False

# info on model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"--- Model Statistics ---")
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Frozen Parameters: {total_params - trainable_params:,}")
print(f"------------------------")

# optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=0.1,
    betas=(0.9, 0.98),
    eps=1e-6
)

# learning rate scheduler
rows_train_data = 0
steps_per_epoch = rows_train_data // (batch_size * accumulation_steps) # how many times gradient is updated per epoch

total_training_steps = steps_per_epoch * epochs
warmup_steps = int(total_training_steps * 0.1) 

scheduler = get_cosine_schedule_with_warmup( # increase linearly and after 10% of total steps starts do decay as a cosine function
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps,
    num_cycles=0.5
)

# checkpoint save function, save weights and info of model and training stage
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, path="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }

    torch.save(checkpoint, path)

    print(f"--- Checkpoint saved at step {step} ---")

# train
def train(): 
    pass

if __name__ == "__main__":
    train()

