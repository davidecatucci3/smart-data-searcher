import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import textwrap
import torch
import os

# prevents memory fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from hyperparameters import hyperparameters
from transformers import BertTokenizer
from import_data import test_loader
from model import Model

# set TF32 instead of FP32
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
batch_size = hyperparameters['batch_size']
d_i = hyperparameters['d_i']
d_t = hyperparameters['d_t']
d_e = hyperparameters['d_e']
max_length_txt = hyperparameters['max_length_txt']

# import model
checkpoint_path = 'checkpoint.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint['model_state_dict']

model = Model().to(device)

model.load_state_dict(state_dict) # load the weights into your model

model.eval()

top_k = 5
res = []
correct_predictions = 0
total_samples = 0

for i, batch in enumerate(test_loader):
    if i == 1: break # limiting to 1 batch for demonstration

    I_batch, T_batch = batch

    I_batch = {k: v.to(device) for k, v in I_batch.items()}
    T_batch = {k: v.to(device) for k, v in T_batch.items()}

    with torch.no_grad():
        logits = model(I_batch, T_batch) 
        labels = torch.arange(logits.size(0), device=device)

        logits_ti = logits.T
        top_probs, topk_indices = logits_ti.topk(top_k, dim=1) # shape: [B, top_k]

        # check if the correct index is ANYWHERE in the top 5
        correct_in_topk = topk_indices.eq(labels.view(-1, 1)).any(dim=1)
        correct_predictions += correct_in_topk.sum().item()
        total_samples += labels.size(0)

        # store the top 5 results for each query in the batch
        for idx in range(logits.size(0)):
            res.append({
                'text': T_batch['input_ids'][idx],
                'truth_image': I_batch['pixel_values'][idx],
                'top_images': [I_batch['pixel_values'][i] for i in topk_indices[idx]],
                'scores': top_probs[idx].tolist()
            })

accuracy = (correct_predictions / total_samples) * 100

print(f'Top-{top_k} Accuracy: {accuracy:.2f}%')

# visualize output 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def visualize_comparison(res_list, num_rows=5):
    _, axes = plt.subplots(num_rows, 6, figsize=(22, num_rows * 4))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def prepare_img(tensor):
        img = tensor.detach().cpu().permute(1, 2, 0).numpy()
        img = std * img + mean
        return np.clip(img, 0, 1)

    for r in range(num_rows):
        item = res_list[r]
        query_text = tokenizer.decode(item['text'], skip_special_tokens=True)

        wrapped_text = "\n".join(textwrap.wrap(query_text, width=25))

        ax_truth = axes[r, 0]
        ax_truth.imshow(prepare_img(item['truth_image']))
        ax_truth.set_title("GROUND TRUTH", color='green', fontweight='bold')
        ax_truth.set_ylabel(wrapped_text, rotation=0, labelpad=80, fontsize=10, va='center')
        ax_truth.set_xticks([])
        ax_truth.set_yticks([])

        for c in range(1, 6):
            img_idx = c - 1
            ax_pred = axes[r, c]
            ax_pred.imshow(prepare_img(item['top_images'][img_idx]))
            
            score = item['scores'][img_idx]
            ax_pred.set_title(f"Rank {c}\n(Score: {score:.3f})", fontsize=9)
            ax_pred.axis('off')

    plt.tight_layout()
    plt.show()

visualize_comparison(res, num_rows=5)

