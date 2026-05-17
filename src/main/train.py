import torch.nn.functional as F
import wandb
import torch
import time
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import get_cosine_schedule_with_warmup
from import_data import train_loader, test_loader
from hyperparameters import hyperparameters
from torch.amp import autocast, GradScaler
from model import Model

# set TF32 instead of FP32 (faster GPU matmul using TF32)
torch.set_float32_matmul_precision('high')

# set device
if torch.cuda.is_available():
    device = torch.device("cuda")

    print(f"Using device: {device}")
else:
    print('No GPU detected!') # THIS CODE WORKS ONLY IF GPU IS AVAILABLE!

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
        "learning_rate":      lr,
        "batch_size":         batch_size,
        "d_i":                d_i,
        "d_t":                d_t,
        "d_e":                d_e,
        "alpha":              alpha,
        "epochs":             epochs,
        "max_test_batches":   max_test_batches,
        "top_k":              top_k,
        "accumulation_steps": accumulation_steps,
        "dataset":            "bitmind/MS-COCO",
        "architecture":       "CLIP",
    }
)

wandb.define_metric("step") # define x-axis metric                
wandb.define_metric("train/*", step_metric="step")
wandb.define_metric("test/*",  step_metric="step")

# set model
model = Model().to(device)  # move parameters to device (GPU)
scaler = GradScaler('cuda') # grad scaler prevent the usage of autocast float16 to underflow by scaling gradient

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
rows_train_data = 590000
steps_per_epoch = rows_train_data // (batch_size * accumulation_steps) # how many times weights are updated per epoch

total_training_steps = steps_per_epoch * epochs
warmup_steps = int(total_training_steps * 0.1) 

scheduler = get_cosine_schedule_with_warmup( # learning rate increase linearly and after 10% of total steps starts do decay as a cosine function
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps,
    num_cycles=0.5
)

# checkpoint save function, save weights and info of model and training stage
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, path="checkpoint.pth"):
    checkpoint = {
        'epoch':                epoch,
        'step':                 step,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict':    scaler.state_dict(),
    }
    torch.save(checkpoint, path)

    print(f"--- Checkpoint saved at step {step} ---")

# train
def train(): 
    # -- TRAINING PHASE --
    train_loss = 0.0
    train_loss_ti = 0.0
    train_loss_it = 0.0

    dt_batch = 0.0

    prev_train_loss = 0.0
    prev_train_loss_ti = 0.0
    prev_train_loss_it = 0.0
    ema_pos_sim = 0.0
    ema_neg_sim = 0.0
    beta = 0.1

    step = 0

    # track model gradients
    wandb.watch(model, log_freq=100)

    time_start = time.time() # track avg step time

    optimizer.zero_grad(set_to_none=True) # set gradient to zero at the starta are not 0

    for epoch in range(epochs):
        for i, batch in enumerate(train_loader): 
            model.train()

            time_start2 = time.time() # track avg batch time

            # I_batch = {"pixel_values": tensor([B, 3, 224, 224])}         
            # T_batch = {"input_ids": ..., "attention_mask": ..., ...}     
            I_batch, T_batch, _, _ = batch 
            
            # put input tensors to device GPU
            I_batch = {k: v.to(device, non_blocking=True) for k, v in I_batch.items()}
            T_batch = {k: v.to(device, non_blocking=True) for k, v in T_batch.items()}

            with autocast(device_type='cuda', dtype=torch.float16): # use float16 instead of float32
                logits = model(I_batch, T_batch) # shape: [B, B], output model

                # calculate loss
                labels = torch.arange(logits.size(0), device=device)
                loss_it = F.cross_entropy(logits, labels)   # image searching for text
                loss_ti = F.cross_entropy(logits.T, labels) # text searching for image

                # alpha pay more attention to loss_i so T->I because I need low loss on that
                local_train_loss = ((1 - alpha)*loss_it + (alpha)*loss_ti) / accumulation_steps

            train_loss += local_train_loss.item() * accumulation_steps
            train_loss_it += loss_it.item()
            train_loss_ti += loss_ti.item()

            # avoid underflow by scaling up loss so gradients don't become to small and after does backpropagation
            scaler.scale(local_train_loss).backward()

            time_end2 = time.time()

            dt_batch += time_end2 - time_start2

            # gradient accumulation technique (update weights each accumulation_steps so to speed up computation by loosing a little bit of precision in gradient updates)
            if i > 0 and i % accumulation_steps == 0:
                with torch.no_grad():
                    # data for positive similarity and negative similarity graph
                    t_scale = torch.exp(model.t).clamp(max=100).item()

                    pos_sim = torch.diag(logits).mean().item() / t_scale # 0 bad, 1 good
                    neg_sim = ((logits.sum() - torch.diag(logits).sum()) / (logits.numel() - logits.size(0))) / t_scale # 0 good, 1 bad 
                    
                    ema_pos_sim = pos_sim*beta + (ema_pos_sim)*(1 - beta)
                    ema_neg_sim = neg_sim*beta + (ema_neg_sim)*(1 - beta)

                    # capture how "big" the gradients are before they vanish
                    total_norm = 0

                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2

                    total_norm = total_norm ** 0.5

                    wandb.log({
                        "train/pos_similarity": ema_pos_sim,
                        "train/neg_similarity": ema_neg_sim,
                        "train/logit_gap":      pos_sim - neg_sim,
                        "train/grad_norm":      total_norm,
                        "step":                 step
                    })
                
                scaler.step(optimizer)                # update parameters
                scaler.update()                       # adjust what did step before
                optimizer.zero_grad(set_to_none=True) # set gradient to zero
                scheduler.step()                      # update learning rate

                # info training
                curr_train_loss = (train_loss - prev_train_loss) / accumulation_steps
                curr_train_loss_it = (train_loss_it - prev_train_loss_it) / accumulation_steps
                curr_train_loss_ti = (train_loss_ti - prev_train_loss_ti) / accumulation_steps

                prev_train_loss = train_loss
                prev_train_loss_ti = train_loss_ti
                prev_train_loss_it = train_loss_it

                dt_avg_batch = dt_batch / (i + 1)
                dt_avg_step = time.time() - time_start

                print(
                    f"step {step}/{total_training_steps} | "
                    f"train loss: {curr_train_loss:.4f} | "
                    f"train loss T->I: {curr_train_loss_ti:.4f} | "
                    f"train loss I->T: {curr_train_loss_it:.4f} | "
                    f"dt step: {dt_avg_step:.4f}s | "
                    f"dt batch: {dt_avg_batch:.4f}s | "
                    f"lr: {optimizer.param_groups[0]['lr']:.8f}"
                )
                
                time_start = time.time()

                # log training metrics
                wandb.log({
                    "train/loss": curr_train_loss,
                    "train/loss_it": curr_train_loss_it,
                    "train/loss_ti": curr_train_loss_ti,
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "step": step
                })

                wandb.log({
                    "train/temp_scaled": torch.exp(model.t).clamp(max=100).item(),
                    "step": step
                })

                # each 25% of tot steps save a checkpoint of current model parameters
                if step % int(total_training_steps * 0.25) == 0:
                    save_checkpoint(model, optimizer, scheduler, scaler, epoch, step)
                
                # -- TEST PHASE --
                if step % int(total_training_steps * 0.05) == 0: # each 5% of the tot steps test the model on unseen data
                    model.eval()
                        
                    test_loss = 0.0
                    test_loss_ti = 0.0
                    test_loss_it = 0.0

                    steps_test = 0

                    correct_predictions = 0.0
                    total_samples = 0

                    dt_avg_batch_test = 0.0      

                    # define WandB table structure to visualize at each test if predictions are correct by visually checking 
                    columns = ["query_text", "ground_truth_image", "top_1_prediction", "top_2_prediction", "top_3_prediction", "top_4_prediction", "top_5_prediction"]
                    test_table = wandb.Table(columns=columns)   

                    with torch.no_grad():     
                        time_x = time.time() # track avg step time

                        for j, batch in enumerate(test_loader):
                            if j == max_test_batches: # 5 * 2816 so test on 15360 pairs
                                break

                            time_start3 = time.time() # track avg batch test time
                        
                            I_batch, T_batch, images, texts = batch

                            I_batch = {k: v.to(device, non_blocking=True) for k, v in I_batch.items()}
                            T_batch = {k: v.to(device, non_blocking=True) for k, v in T_batch.items()}

                            with autocast(device_type='cuda', dtype=torch.float16): # use FP16/BF16 instead of FP32
                                logits = model(I_batch, T_batch) 

                                # visualize prediction model in WandB to see if it's predicting correctly 
                                if j == 0:                     
                                    # get the top-5 indices for each image in the batch
                                    logits_ti = logits.T
                                    _, topk_indices = logits_ti.topk(top_k, dim=1) 
                                    
                                    for idx in range(min(10, len(texts))):
                                        query_text = texts[idx]
                                        gt_image = wandb.Image(images[idx])
                                            
                                        # get the 5 images the model liked most for this text
                                        p1 = wandb.Image(images[topk_indices[idx][0].item()])
                                        p2 = wandb.Image(images[topk_indices[idx][1].item()])
                                        p3 = wandb.Image(images[topk_indices[idx][2].item()])
                                        p4 = wandb.Image(images[topk_indices[idx][3].item()])
                                        p5 = wandb.Image(images[topk_indices[idx][4].item()])
                                            
                                        # add the row to the table
                                        test_table.add_data(query_text, gt_image, p1, p2, p3, p4, p5)
                                        
                                    wandb.log({"eval/text_to_image_search": test_table, "global_step": step})
                                
                                # dynamic labels based on the actual batch size
                                labels = torch.arange(logits.size(0), device=device)

                                # calculate loss
                                loss_it = F.cross_entropy(logits, labels)
                                loss_ti = F.cross_entropy(logits.T, labels)
                                local_test_loss = (1 - alpha)*loss_it + (alpha)*loss_ti

                                # calculate accuracy, we look at logits col because columns of logits are texts, rows are images
                                logits_ti = logits.T
                                _, topk_indices = logits_ti.topk(top_k, dim=1) # shape: [B, top_k]

                                correct_in_topk = topk_indices.eq(labels.view(-1, 1)).any(dim=1)

                                correct_predictions += correct_in_topk.sum().item()
                                total_samples += labels.size(0)
                            
                            test_loss += local_test_loss.item()
                            test_loss_it += loss_it.item()
                            test_loss_ti += loss_ti.item()
                            steps_test += 1

                            dt_avg_batch_test += time.time() - time_start3
                        
                        dt_step_test = time.time() - time_x

                        # print accuracy and test loss
                        test_loss = test_loss / steps_test
                        test_loss_it = test_loss_it / steps_test
                        test_loss_ti = test_loss_ti / steps_test

                        accuracy = (correct_predictions / total_samples) * 100

                        print(
                            f"step {step}/{total_training_steps} | "
                            f"test loss: {test_loss:.4f} | "
                            f"test loss T->I: {test_loss_ti:.4f} | "
                            f"test loss I->T: {test_loss_it:.4f} | "
                            f"accuracy: {accuracy:.2f}%"
                        )

                        # log test metrics
                        wandb.log({
                            "test/loss":     test_loss,
                            "test/loss_it":  test_loss_it,
                            "test/loss_ti":  test_loss_ti,
                            "test/accuracy": accuracy,
                            "step":   step
                        })

                        wandb.log({
                            "test/dt_step":  dt_step_test,
                            "test/dt_batch": dt_avg_batch_test / max_test_batches,
                            "step":   step
                        })

                step += 1

if __name__ == "__main__":
    train()

