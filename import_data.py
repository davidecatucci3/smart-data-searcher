import io

from transformers import ViTImageProcessor, BertTokenizer
from hyperparameters import hyperparameters
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image

# hyperparameters
batch_size = hyperparameters['batch_size']
max_length_txt = hyperparameters['max_length_txt']

# import dataset
dataset = load_dataset("bitmind/MS-COCO", streaming=True)

# import pre-trained models
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
bert_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
 
def collate_fn(batch):
    images = [] # list of images in PIL objects form

    for item in batch:
        img_data = item['image']

        # if streaming returns a dict instead of a PIL object the data arrives as bytes and bot as PIL object so i need to convert it to PIL object
        if isinstance(img_data, dict):
            if 'bytes' in img_data and img_data['bytes'] is not None:
                img = Image.open(io.BytesIO(img_data['bytes']))
            elif 'path' in img_data:
                img = Image.open(img_data['path'])
            else:
                # fallback for unexpected dict structures
                img = Image.new('RGB', (224, 224), color='black')
        else:
            img = img_data # it's already a PIL image

        # convert all images to RGB (fixes grayscale/RGBA issues)
        images.append(img.convert("RGB"))

    texts = [item['sentences']['raw'][0] if isinstance(item['sentences']['raw'], list) else item['sentences']['raw'] for item in batch] # list of texts not tokens

    # process images so return a list of images in tensor / array form
    img_inputs = vit_processor(images=images, return_tensors="pt")

    # process text so return list texts converted in tokens 
    txt_inputs = bert_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length_txt)

    return img_inputs, txt_inputs, images, texts

train_data = dataset['train'].shuffle(seed=42, buffer_size=10000)
test_data = dataset['test'].shuffle(seed=42, buffer_size=10000).filter(lambda _, i: i % 5 == 0, with_indices=True) # in test data each 5 sample the image is the same and change only the text, so now i am taking only 1 image each 5 for testing 

train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size * 12, collate_fn=collate_fn, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2, shuffle=False)
