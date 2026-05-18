from google.colab import drive

drive.mount('/content/drive')

import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import textwrap
import torch
import math
import os
import io

# prevents memory fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from transformers import ViTImageProcessor, BertTokenizer
from transformers import ViTModel, BertModel
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image

# set TF32 instead of FP32 (faster GPU matmul using TF32)
torch.set_float32_matmul_precision('high')

# set device (RECOMENDED INFERENCE USING GPU BUT ALSO CPU AND MPS USABLE)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# hyperparameters   
hyperparameters = {
    'batch_size': 4,
    'd_i': 768,             # dimension vector text encoder
    'd_t': 768,             # dimension vector image encoder
    'd_e': 512,              # dimension vector joint embedding
    'max_test_batches': 5,  # maximum number of batches to elaborate for testing
    'lr': 1e-4,
    'epochs': 8,
    'accumulation_steps': 8, # each how many steps weights are updated
    'max_length_txt': 128,   # max length raw text for text encoder
    'alpha': 0.8,
    'top_k': 5               # number of images selected from search (the top 5)
} 

batch_size = hyperparameters['batch_size']
d_i = hyperparameters['d_i']
d_t = hyperparameters['d_t']
d_e = hyperparameters['d_e']
max_length_txt = hyperparameters['max_length_txt']

# import dataset
dataset = load_dataset("bitmind/MS-COCO", streaming=True) # we don't download dataset on local but we use streaming to iterate on the dataset on the fly without download dataset

# import pre-trained models
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k") # image encoder
bert_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")        # text encoder
 
def collate_fn(batch):
    images = [] # list of images in PIL objects form

    for item in batch:
        img_data = item['image']

        # due to streaming it can return both the PIL object correct or sometimes a dict with inside raw bytes or path of image
        if isinstance(img_data, dict):
            if 'bytes' in img_data and img_data['bytes'] is not None: # if dict contains raw bytes
                img = Image.open(io.BytesIO(img_data['bytes']))
            elif 'path' in img_data: # if dict contains path
                img = Image.open(img_data['path'])
            else: # else
                # fallback for unexpected dict structures
                img = Image.new('RGB', (224, 224), color='black')
        else:
            img = img_data # it's already a PIL image

        # convert all images to RGB (fixes grayscale/RGBA issues)
        images.append(img.convert("RGB"))

    # list of texts not tokens
    texts = [item['sentences']['raw'][0] if isinstance(item['sentences']['raw'], list) else item['sentences']['raw'] for item in batch] 

    # process images so return a list of images in tensor / array form -> {"pixel_values": tensor([B, 3, 224, 224])}        
    img_inputs = vit_processor(images=images, return_tensors="pt")

    # process text so return list texts converted in tokens -> {"input_ids": ..., "attention_mask": ..., ...}     
    txt_inputs = bert_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length_txt)

    return img_inputs, txt_inputs, images, texts

def filter_every_fifth(_, i):
    return i % 5 == 0

train_data = dataset['train'].shuffle(seed=42, buffer_size=10000) # shuffle to prevent model to learn sequence of data (put in memory only 10.000 and sample randomly from there, when one batch is taken out another is take in)
test_data = dataset['test'].shuffle(seed=42, buffer_size=10000).filter(filter_every_fifth, with_indices=True) # in test data each 5 sample the image is the same and change only the text, so now i am taking only 1 image each 5 for testing 

train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2, shuffle=False)

# model
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # import pre-trained models for text encoder and image encoder
        self.img_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", attn_implementation="sdpa") # sdpa is a variant of flash attention
        self.txt_encoder = BertModel.from_pretrained("google-bert/bert-base-uncased", attn_implementation="sdpa")

        # save memory by not storing all gradients but this is slower due to recalculation after (USE ONLY IF YOU DON'T HAVE ENOUGH MEMORY)
        #self.img_encoder.gradient_checkpointing_enable()
        #self.txt_encoder.gradient_checkpointing_enable()
        
        # parameters
        self.W_i = nn.Parameter(torch.empty(d_i, d_e)) 
        self.W_t = nn.Parameter(torch.empty(d_t, d_e)) 
        self.t = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
        
        # truncated normal initialization parameters (taken from OpenAI paper CLIP) -> sample weights from a normal distribution ~ Normal(mean=0, sd=1 / sqrt(d_in))
        std_wi = 1 / math.sqrt(d_i)
        std_wt = 1 / math.sqrt(d_t)

        torch.nn.init.trunc_normal_(self.W_i, std=std_wi, a=-2*std_wi, b=2*std_wi) # bounded between a and b
        torch.nn.init.trunc_normal_(self.W_t, std=std_wt, a=-2*std_wt, b=2*std_wt)

    def forward(self, I_batch, T_batch):
        # I_batch shape: [B, C, H, W]
        # T_batch shape: [B, max_length_txt]

        I_f = self.img_encoder(**I_batch).last_hidden_state # shape: [B, C*H*W, d_i]
        T_f = self.txt_encoder(**T_batch).last_hidden_state # shape: [B, max_length_txt, d_t]

        I_f = I_f[:, 0, :] # shape: [B, d_i], 0 is the [CLS] token that capture the global meaning of the whole sequence
        T_f = T_f[:, 0, :] # shape: [B, d_t]

        I_e = F.normalize(I_f @ self.W_i, p=2, dim=1) # shape: [B, d_e] -> normalize (makes it norm 1): rowi /= ||rowi||2
        T_e = F.normalize(T_f @ self.W_t, p=2, dim=1) # shape: [B, d_e] -> normalize (makes it norm 1): rowi /= ||rowi||2

        t_scale = torch.exp(self.t).clamp(max=100) # scalar that that controls similarity sharpness globally, clamp prevent t to explode

        logits = I_e @ T_e.T * t_scale # shape: [B, B], similarity matrix: just I_e @ T_e.T not divided by their norms because norms are 1

        return logits

# import model
checkpoint_path = '/content/drive/MyDrive/checkpoint.pth' # trained weights file

checkpoint_path = '/content/drive/MyDrive/checkpoint.pth'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Missing '{checkpoint_path}'. Please upload your weights file to Colab.")

checkpoint = torch.load(checkpoint_path, map_location=device)   
state_dict = checkpoint['model_state_dict']                    

model = Model().to(device)

model.load_state_dict(state_dict) # load the trained weights into your model

model.eval() # evaluation / inference mode

top_k = 3 # top highest probabilities images selected corresponding to the text prompt
res = []  # for each sample {'text': prompt 'truth_image': image, 'top_images': list of top k images, 'scores': probabilities for each top k images find}
correct_predictions = 0
total_samples = 0

for i, batch in enumerate(test_loader):
        if i == 1: break # limiting to 1 batch for demonstration

        I_batch, T_batch, _, _ = batch

        I_batch = {k: v.to(device) for k, v in I_batch.items()}
        T_batch = {k: v.to(device) for k, v in T_batch.items()}

        with torch.no_grad():
            # output model
            logits = model(I_batch, T_batch) 

            # top k images find
            logits_ti = logits.T
            top_probs, topk_indices = logits_ti.topk(top_k, dim=1) # shape: [B, top_k]

            # check if the correct index is anywhere in the top 5
            labels = torch.arange(logits.size(0), device=device)
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

# visualize predictions 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def prepare_img(tensor): # the images are standardized and normalized for more efficinet computation and geometrical needs so we need to do some manipulation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = std * img + mean
            
    return np.clip(img, 0, 1)

def visualize(res, num_rows=2):
    _, axes = plt.subplots(num_rows, 6, figsize=(22, num_rows * 4))

    for i in range(num_rows):
        item = res[i]
        query_text = tokenizer.decode(item['text'], skip_special_tokens=True)

        wrapped_text = '\n'.join(textwrap.wrap(query_text, width=25))

        ax_truth = axes[i, 0]
        ax_truth.imshow(prepare_img(item['truth_image']))
        ax_truth.set_title("GROUND TRUTH", color='green', fontweight='bold')
        ax_truth.set_ylabel(wrapped_text, rotation=0, labelpad=80, fontsize=10, va='center')
        ax_truth.set_xticks([])
        ax_truth.set_yticks([])

        for j in range(top_k):
            ax_pred = axes[i, j]
            ax_pred.imshow(prepare_img(item['top_images'][j]))
                
            score = item['scores'][j]
            ax_pred.set_title(f'Rank {j} \n (Score: {score:.3f})', fontsize=9)
            ax_pred.axis('off')

    plt.tight_layout()
    plt.show()

visualize(res, num_rows=5)