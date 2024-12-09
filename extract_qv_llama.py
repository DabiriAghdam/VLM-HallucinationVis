from collections import OrderedDict
import torch
import numpy as np
from transformers import LlavaProcessor, LlavaForConditionalGeneration, AutoTokenizer, CLIPImageProcessor, AutoProcessor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from PIL import Image
import os
import requests
import base64
import json
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from scipy.stats import mode
import torch.nn as nn

from functools import partial
from tqdm import tqdm as std_tqdm

from transformers.models.llama import shared_state
from transformers.cache_utils import Cache, DynamicCache, StaticCache

tqdm = partial(std_tqdm, dynamic_ncols=True)

torch.set_printoptions(precision=3)

def create_output_dirs():
    os.makedirs("original_images", exist_ok=True)
    os.makedirs("llava_image_patches", exist_ok=True)
    os.makedirs("llava_attention", exist_ok=True)

def convert_image_to_base64(filepath):
    binary_fc = open(filepath, 'rb').read()
    base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')
    ext = filepath.split('.')[-1]
    return f'data:image/{ext};base64,{base64_utf8_str}'

def clear_gpu_memory():
    """Helper to aggressively clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
      

# Load the Llava model and processor
model_name = "llava-hf/llava-1.5-7b-hf"

# Initialize model with proper configuration
model = LlavaForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 to save memory
    device_map="auto",  # Automatically handle device placement
    # low_cpu_mem_usage=True,
    # load_in_8bit=True,  # Enable 8-bit quantization
    # attn_implementation="eager"
)

# Enable memory efficient attention and gradient checkpointing
# model.config.use_memory_efficient_attention = True
# model.gradient_checkpointing_enable()

num_images = 2
patch_size = 14 
border_width = int(patch_size / 8)

centering = True
scale = False
num_layers = model.config.text_config.num_hidden_layers
num_heads = model.config.text_config.num_attention_heads
head_dim = model.config.text_config.hidden_size // num_heads
image_token_index = model.config.image_token_index

# Load and initialize tokenizer and image processor
tokenizer = AutoTokenizer.from_pretrained(model_name)
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)
processor.vision_feature_select_strategy = "default"
processor.patch_size = patch_size

# Initialize vision tower
model.vision_tower.to(model.device)
model.eval()

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Single image and text
urls = ["https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg", 
        "http://images.cocodataset.org/val2017/000000039769.jpg"]

text = "USER: \ndescribe the following image <image>. ASSISTANT:"

all_output = {}
all_attentions = {}
all_patches = None
all_images = []
all_sentences = []
all_token_ids = []
all_queries = {}
all_keys = {}

for i in tqdm(range(num_images)):
    clear_gpu_memory()

    # Load the image
    image = Image.open(requests.get(urls[i], stream=True).raw)

    # Create directory for original images
    create_output_dirs()

    # Process the image and text
    inputs = processor(images=image, text=text, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device, torch.float16)

    # Run the model
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=20, use_cache=True, return_dict_in_generate=True, output_attentions=True)

    all_sentences.append(processor.batch_decode(outputs['sequences'], skip_special_tokens=True)[0].strip())
    all_token_ids.append(outputs['sequences'][0][:-1])

    attentions = outputs.attentions

    num_layers = len(attentions[0])  # Number of layers
    batch_size, num_heads, M, _ = attentions[0][0].shape 
    final_len = attentions[-1][0].shape[-1]

    full_attentions = []
    for layer in range(num_layers):

        full_attention = torch.zeros(batch_size, num_heads, final_len, final_len, \
            dtype=attentions[0][layer].dtype, device=attentions[0][layer].device)

        full_attention[:, :, :M, :M] = attentions[0][layer]

        # Now, fill in each subsequent row from the remaining attentions.
        # attentions[i] for i>0 has shape [1, num_heads, 1, M+i]
        # That corresponds to the (M + (i-1))th row in the final matrix.
        for vector_index in range(1, len(attentions)):
            # This is one new row (the i-th generated token, zero-based index)
            row_index = M + (vector_index - 1) 
            # attentions[i][0] has shape [num_heads, 1, M+i], we want to place it at:
            # full_attention[:, :, row_index, :M+i] = that row
            full_attention[:, :, row_index:row_index+1, :M+vector_index] = attentions[vector_index][layer]

        # Now 'full_attention' contains a [batch_size, num_heads, final_len, final_len] for the model.
        full_attentions.append(full_attention)

    all_attentions[i] = full_attentions

    # Preprocess the image for patch extraction
    np_image = inputs.pixel_values[0].permute(1, 2, 0).cpu().numpy()
    all_images.append(np_image)
    np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())

    # Save processed image
    filename_prefix = urls[i][urls[i].rfind("/") + 1:urls[i].rfind(".")]
    plt.imsave(f"original_images/original_image_{filename_prefix}.png", np_image)

    # Calculate the number of patches
    h, w, _ = np_image.shape
    h_patches = h // patch_size
    w_patches = w // patch_size

    # Extract image patches
    image_patches = np_image.reshape(h_patches, patch_size, w_patches, patch_size, 3)
    image_patches = image_patches.swapaxes(1, 2)
    image_patches = image_patches.reshape(h_patches * w_patches, patch_size, patch_size, 3)

    if (i == 0):
        all_patches = image_patches
    else:
        all_patches = np.concatenate([all_patches, image_patches])

    pink_border_h = np.ones(shape=(patch_size, border_width, 3)) * np.array([227 / 255, 55 / 255, 143 / 255]).reshape(-1, 3)
    pink_border_v = np.ones(shape=(border_width, patch_size + border_width * 2, 3)) * np.array([227 / 255, 55 / 255, 143 / 255]).reshape(-1, 3)
    green_border_h = np.ones(shape=(patch_size, border_width, 3)) * np.array([95 / 255, 185 / 255, 108 / 255]).reshape(-1, 3)
    green_border_v = np.ones(shape=(border_width, patch_size + border_width * 2, 3)) * np.array([95 / 255, 185 / 255, 108 / 255]).reshape(-1, 3)
    pink_border_h = np.repeat(np.expand_dims(pink_border_h, 0), (h // patch_size) ** 2, axis=0)
    pink_border_v = np.repeat(np.expand_dims(pink_border_v, 0), (h // patch_size) ** 2, axis=0)
    green_border_h = np.repeat(np.expand_dims(green_border_h, 0), (h // patch_size) ** 2, axis=0)
    green_border_v = np.repeat(np.expand_dims(green_border_v, 0), (h // patch_size) ** 2, axis=0)

    for layer in range(num_layers):
        if i == 0:
            all_queries[layer] = {}
            all_keys[layer] = {}

        query_heads = shared_state.query_key[layer][0]
        key_heads = shared_state.query_key[layer][1]

        for head in range(num_heads):
            if (i == 0):
                all_queries[layer][head] = query_heads[:, head, :, :].cpu().detach().numpy()[0, :]
                all_keys[layer][head] = key_heads[:, head, :, :].cpu().detach().numpy()[0, :]
            else:
                all_queries[layer][head] = np.concatenate((all_queries[layer][head], query_heads[:, head, :, :].cpu().detach().numpy()[0, :]), axis=0)
                all_keys[layer][head] = np.concatenate((all_keys[layer][head], key_heads[:, head, :, :].cpu().detach().numpy()[0, :]), axis=0)
    key_image_patches = np.concatenate([pink_border_v, 
                                        np.concatenate([pink_border_h, image_patches, pink_border_h], axis=2),
                                        pink_border_v], axis=1)

    query_image_patches = np.concatenate([green_border_v, 
                                            np.concatenate([green_border_h, image_patches, green_border_h], axis=2),
                                            green_border_v], axis=1)

    for nth_image in range(image_patches.shape[0]):
        plt.imsave(f"llava_image_patches/{filename_prefix}_patch_{nth_image}.png", image_patches[nth_image])
        plt.imsave(f"llava_image_patches/key_{filename_prefix}_patch_{nth_image}.png", key_image_patches[nth_image])
        plt.imsave(f"llava_image_patches/query_{filename_prefix}_patch_{nth_image}.png", query_image_patches[nth_image])
    
    shared_state.query_key = DynamicCache()

if centering:
    queries = all_queries[layer][head].copy().astype("float")
    keys = all_keys[layer][head].copy().astype("float")
    mean_shift = np.mean(queries, axis=0) - np.mean(keys, axis=0)
    keys += mean_shift
    all_keys[layer][head] = keys

if scale:
    queries = all_queries[layer][head].copy().astype("float")
    keys = all_keys[layer][head].copy().astype("float")
    q_n = np.linalg.norm(queries, axis=1).mean()
    k_n = np.linalg.norm(keys, axis=1).mean()
    
    c = np.sqrt(q_n / k_n) 

    keys *= c
    queries *= (1 / c)
    all_keys[layer][head] = keys
    all_queries[layer][head] = queries

last_start = 0
last_end = 0
for i in range(num_images):
    start = len(all_token_ids[i-1]) if i >= 1 else 0
    end = len(all_token_ids[i])

    last_start += start
    last_end += end
    print(last_start, last_end)

    for layer in range(num_layers):
            if i == 0:
                all_output[layer] = {}

            for head in range(num_heads):
                combined = np.concatenate([all_queries[layer][head][last_start:last_end], all_keys[layer][head][last_start:last_end]])
                if (i == 0):
                    all_output[layer][head] = combined
                else:
                    all_output[layer][head] = np.concatenate([all_output[layer][head], combined])

# Perform dimensionality reduction
llava_embeddeds = {"PCA": {}, "TSNE": {}, "UMAP": {}, "PCA_3d": {}, "TSNE_3d": {}, "UMAP_3d": {}}
for layer in tqdm(range(num_layers)):
    llava_embeddeds["TSNE"][layer] = {}
    llava_embeddeds["TSNE_3d"][layer] = {}
    llava_embeddeds["PCA"][layer] = {}
    llava_embeddeds["PCA_3d"][layer] = {}
    llava_embeddeds["UMAP"][layer] = {}
    llava_embeddeds["UMAP_3d"][layer] = {}
    for head in (range(num_heads)):
        # llava_embeddeds["TSNE"][layer][head] = TSNE(n_components=2, learning_rate='auto', n_jobs=-1, metric="cosine",
        #                                           init='random', perplexity=20).fit_transform(all_output[layer][head])
        # llava_embeddeds["TSNE_3d"][layer][head] = TSNE(n_components=3, learning_rate='auto', n_jobs=-1, metric="cosine",
        #                                              init='random', perplexity=20).fit_transform(all_output[layer][head])
        
        llava_embeddeds["PCA"][layer][head] = PCA(n_components=2,).fit_transform(all_output[layer][head])
        llava_embeddeds["PCA_3d"][layer][head] = PCA(n_components=3,).fit_transform(all_output[layer][head])
        
        # reducer = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.1, metric="cosine",)
        # llava_embeddeds["UMAP"][layer][head] = reducer.fit_transform(all_output[layer][head])
        
        # reducer = umap.UMAP(n_neighbors=10, n_components=3, min_dist=0.1, metric="cosine",)
        # llava_embeddeds["UMAP_3d"][layer][head] = reducer.fit_transform(all_output[layer][head])

# Load DeepLabV3 model and weights
weights = DeepLabV3_ResNet50_Weights.DEFAULT
transforms = weights.transforms()#(resize_size=[(336, 336)])
model_seg = deeplabv3_resnet50(weights=weights, progress=False)
model_seg = model_seg.eval()
batch = torch.stack([transforms(Image.fromarray(all_images[i].astype('uint8'))) for i in range(num_images)])
output = model_seg(batch)['out']
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
sem_idx_to_class = {sem_class_to_idx[key]: key for key in sem_class_to_idx}
sem_idx_to_class[0] = "bg"
semantic_labels = {}
for nth_image in range(num_images):
    semantic_labels[nth_image] = []
    seg = torch.argmax(output[nth_image], dim=0).cpu().detach().numpy()
    for i in range(0, 336, patch_size):
        for j in range(0, 336, patch_size):
            label = sem_idx_to_class[mode(seg[i: i + patch_size, j: j + patch_size].ravel())[0][0]]
            semantic_labels[nth_image].append(label)

# After saving embeddings, add JSON creation
os.makedirs(f"llava_layer", exist_ok=True)
# Create embedding JSONs
for layer in tqdm(range(num_layers)):
    for head in range(num_heads):
        embedding_json = {
            "layer": layer,
            "head": head,
            "tokens": []
        }
        for i in range(len(llava_embeddeds["PCA"][layer][head])):
            token_data = {
                "tsne_x": round(float(llava_embeddeds["PCA"][layer][head][i, 0]), 3),
                "tsne_y": round(float(llava_embeddeds["PCA"][layer][head][i, 1]), 3),
                "tsne_x_3d": round(float(llava_embeddeds["PCA_3d"][layer][head][i, 0]), 3),
                "tsne_y_3d": round(float(llava_embeddeds["PCA_3d"][layer][head][i, 1]), 3),
                "tsne_z_3d": round(float(llava_embeddeds["PCA_3d"][layer][head][i, 2]), 3),
                "pca_x": round(float(llava_embeddeds["PCA"][layer][head][i, 0]), 3),
                "pca_y": round(float(llava_embeddeds["PCA"][layer][head][i, 1]), 3),
                "pca_x_3d": round(float(llava_embeddeds["PCA_3d"][layer][head][i, 0]), 3),
                "pca_y_3d": round(float(llava_embeddeds["PCA_3d"][layer][head][i, 1]), 3),
                "pca_z_3d": round(float(llava_embeddeds["PCA_3d"][layer][head][i, 2]), 3),
                "umap_x": round(float(llava_embeddeds["PCA"][layer][head][i, 0]), 3),
                "umap_y": round(float(llava_embeddeds["PCA"][layer][head][i, 1]), 3),
                "umap_x_3d": round(float(llava_embeddeds["PCA_3d"][layer][head][i, 0]), 3),
                "umap_y_3d": round(float(llava_embeddeds["PCA_3d"][layer][head][i, 1]), 3),
                "umap_z_3d": round(float(llava_embeddeds["PCA_3d"][layer][head][i, 2]), 3)
            }
            embedding_json["tokens"].append(token_data)
        
        with open(f"llava_layer/layer{layer}_head{head}.json", "w") as f:
            json.dump(embedding_json, f)

        attention_json = {
            "layer": layer,
            "head": head,
            "tokens": []
        }
        
        for i in range(num_images):
            sel_attention = all_attentions[i][layer][0][head].clone()
            
            # Add row-wise attention
            for j in range(len(sel_attention)):
                attention_json["tokens"].append({"attention": [round(float(val), 3) for val in sel_attention[j].tolist()]})
            
            # Add column-wise attention
            for j in range(sel_attention.shape[1]):
                attention_json["tokens"].append({"attention": [round(float(val), 3) for val in sel_attention[:, j].tolist()]})
            
        # Save attention JSON
        with open(f"llava_attention/layer{layer}_head{head}.json", "w") as outfile:
            json.dump(attention_json, outfile)

def process_text_token(token_type, pos_int, position, token_id, sentence):
    """Process a text token and return its token data"""
    return {
        "value": tokenizer.decode([token_id]).strip(),
        "type": token_type,
        "pos_int": pos_int,
        "length": len(sentence),
        "position": position,
        "sentence": sentence,
    }

def process_image_token(token_type, i, filename_prefix, patch_size, semantic_labels, nth_data):
    """Process an image token and return its token data"""
    
    dataurl = convert_image_to_base64(f"llava_image_patches/{token_type}_{filename_prefix}_patch_{i}.png")
    original_patch_dataurl = dataurl
    original_image_dataurl = "null"
    
    row = i // (336 // patch_size)
    col = i % (336 // patch_size)
    ad_row = row
    ad_col = col
    
    return {
        "originalImagePath": original_image_dataurl,
        "originalPatchPath": original_patch_dataurl,
        "imagePath": dataurl,
        "position": row,
        "pos_int": col,
        "position_row": ad_row,
        "position_col": ad_col,
        "type": token_type,
        "value": semantic_labels[nth_data][i % ((336 // patch_size) ** 2 + 1)]
    }

token_json = {"tokens": []}
for nth_data in range(num_images):
    filename_prefix = urls[nth_data][urls[nth_data].rfind("/") + 1:urls[nth_data].rfind(".")]
    
    token_ids = all_token_ids[nth_data]

    for token_type in ["query", "key"]:
        position = 0 #???
        i = 0
        for token_idx in range(len(token_ids)):
            if (token_ids[token_idx] == image_token_index):
                    token_data = process_image_token(token_type, i, filename_prefix, patch_size, semantic_labels, nth_data)
                    token_json["tokens"].append(token_data)
                    i += 1
            else: #336 / ... + 1 ??
                token_data = process_text_token(token_type, position, position / (len(token_ids) - 1 - (336 / patch_size) ** 2 + 1), token_ids[token_idx], all_sentences[nth_data])
                token_json["tokens"].append(token_data)
                position += 1

with open(f"tokens.json", "w") as f:
    json.dump(token_json, f)
