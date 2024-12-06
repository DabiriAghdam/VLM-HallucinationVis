from collections import OrderedDict
import torch
import numpy as np
# from tqdm import tqdm
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
tqdm = partial(std_tqdm, dynamic_ncols=True)

torch.set_printoptions(precision=3)

class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = []
    def hook_fn(self, module, input, output):
        self.features.append(output.detach())
    def close(self):
        self.hook.remove()

def create_output_dirs():
    os.makedirs("original_images", exist_ok=True)
    os.makedirs("llava_image_patches", exist_ok=True)
    os.makedirs("llava_attention", exist_ok=True)
    # os.makedirs("llava_embeddings", exist_ok=True)

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
# Adjust the bias term (Out of date, don't use)
adjust_bias_term = False
centering = False
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
# processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
# Initialize vision tower
model.vision_tower.to(model.device)
model.eval()  # Set to evaluation mode

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
for i in tqdm(range(num_images)):
    clear_gpu_memory()
      # Initialize the hook
    features = OrderedDict()
    # Set up hooks to extract query and key features
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ("q_proj" in name or "k_proj" in name):
            features[name] = ModuleHook(module)

    # Load the image
    image = Image.open(requests.get(urls[i], stream=True).raw)

    # Create directory for original images
    create_output_dirs()

    # Process the single image
    inputs = image_processor(image, return_tensors="pt")

    # Process the image and text
    inputs = processor(images=image, text=text, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device, torch.float16)
    # Run the model
    with torch.inference_mode():
        # outputs = model(**inputs, output_attentions=False)
        outputs = model.generate(**inputs, max_new_tokens=20, use_cache=True, past_key_values=None, return_dict_in_generate=True, output_hidden_states=True, output_attentions=True)
        # clear_gpu_memory()
    # Close hooks
    for feature in features.values():
        feature.close()
    print(outputs)
    an
    print(([attn[0].shape for attn in outputs.attentions]))
    all_sentences.append(processor.batch_decode(outputs['sequences'], skip_special_tokens=True)[0].strip())
    print(processor.batch_decode(outputs['sequences'], skip_special_tokens=True)[0].strip())
    all_token_ids.append(outputs['sequences'][0])

    # all_attentions[i] = outputs.attentions[-1]

    # Assuming outputs.attentions is a list of tensors with varying sizes
    attentions = outputs.attentions
    # Initialize the list to store the attention matrices for each layer
    attention_matrices = [[] for _ in range(len(attentions[0]))]
    # Process the self-attention for the input tokens
    for layer_idx in range(len(attentions[0])):
        self_attention = attentions[0][layer_idx].squeeze(0)
        attention_matrices[layer_idx].append(self_attention)
    # Process the attention for the generated tokens
    for attn in attentions[1:]:
        for layer_idx in range(len(attn)):
            attention_matrices[layer_idx].append(attn[layer_idx].squeeze(0))
    # Concatenate the attention matrices along the token dimension for each layer
    all_attention_matrices = []
    for layer_attns in attention_matrices:
        # Find the maximum size along the token dimension
        max_size = max(attn.size(-1) for attn in layer_attns)
        # Pad the tensors to match the maximum size
        padded_attns = []
        for attn in layer_attns:
            pad_size = max_size - attn.size(-1)
            padded_attn = torch.nn.functional.pad(attn, (0, pad_size))
            padded_attns.append(padded_attn)
        # Concatenate the padded tensors along the token dimension
        all_attention_matrices.append(torch.cat(padded_attns, dim=1))
    # Now all_attention_matrices should contain the attention matrices for each layer
    # for idx, attn_matrix in enumerate(all_attention_matrices):
    #     print(f"Layer {idx} attention matrix shape: {attn_matrix.shape}")  # Should print torch.Size([32, 613, 613])
    #is this correct?
    # all_attentions[i] = all_attention_matrices

    # Preprocess the image for patch extraction
    np_image = inputs.pixel_values[0].permute(1, 2, 0).cpu().numpy()
    all_images.append(np_image)
    np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())
    # np_image /= 2
    # np_image += 0.5

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

    # Extract and process features
    attentions = {}
    for layer in range(num_layers):
        if i == 0:
            all_output[layer] = {}
        attentions[layer] = {}
        layer_query_name = f"language_model.model.layers.{layer}.self_attn.q_proj"
        layer_key_name = f"language_model.model.layers.{layer}.self_attn.k_proj"
        if adjust_bias_term and False:
            raw_query_feature = features[layer_query_name].features[0].cpu().detach().numpy()[0, :] - model.encoder.layer[layer].attention.attention.query.bias.clone().cpu().detach().numpy()
            raw_key_feature = features[layer_key_name].features[0].cpu().detach().numpy()[0, :] - model.encoder.layer[layer].attention.attention.key.bias.clone().cpu().detach().numpy()
        else:
            #is it correct?
            concatenated_hidden_states = torch.cat([step_hidden_states[layer + 1] for step_hidden_states in outputs['hidden_states']], dim=1)
            raw_query_feature = model.language_model.model.layers[layer].self_attn.q_proj(concatenated_hidden_states) #features[layer_query_name].features[0].cpu().detach().numpy()[0, :]
            raw_key_feature = outputs['past_key_values'][layer][0].permute(0, 2, 1, 3).contiguous().view(1, outputs['past_key_values'][layer][0].size(2), -1) #features[layer_key_name].features[0].cpu().detach().numpy()[0, :]
            #.contiguous()?
        # query_heads = torch.split(features[layer_query_name].features[0], features[layer_query_name].features[0].size(-1) // num_heads, dim=-1)
        # key_heads = torch.split(features[layer_key_name].features[0], features[layer_key_name].features[0].size(-1) // num_heads, dim=-1)
        query_heads = torch.split(raw_query_feature, raw_query_feature.size(-1) // num_heads, dim=-1)
        key_heads = torch.split(raw_key_feature, raw_key_feature.size(-1) // num_heads, dim=-1)
        # print(query_heads[0].shape)
        # print(key_heads[0].shape)

        list_tensor = []
        for head in range(num_heads):
            list_tensor.append(nn.functional.softmax(torch.bmm(query_heads[head], key_heads[head].transpose(1, 2)) / np.sqrt(head_dim), dim=-1))
        attentions[layer] = torch.stack(list_tensor, dim=1)
        
        for head in range(num_heads):
            combined = np.concatenate([query_heads[head].cpu().detach().numpy()[0, :], key_heads[head].cpu().detach().numpy()[0, :]])
            if (i == 0):
                all_output[layer][head] = combined
            else:
                all_output[layer][head] = np.concatenate([all_output[layer][head], combined])

    all_attentions[i] = attentions
    print(attentions[0].shape)

    # for idx, attn_matrix in enumerate(all_attention_matrices):
    #     print(attn_matrix == all_attentions[i][idx]) 

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
    semantic_labels[nth_image] = ["CLS"]
    seg = torch.argmax(output[nth_image], dim=0).cpu().detach().numpy()
    for i in range(0, 336, patch_size):
        for j in range(0, 336, patch_size):
            label = sem_idx_to_class[mode(seg[i: i + patch_size, j: j + patch_size].ravel())[0][0]]
            semantic_labels[nth_image].append(label)

# Save segmentation image instead of displaying
# plt.figure()
# plt.title("Segmentation Result")
# plt.imshow(seg)
# plt.savefig("segmentation_result.png")
# plt.close()

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
    if i == 0:
        dataurl = f"https://raw.githubusercontent.com/catherinesyeh/attention-viz/VIT-vis/img/cls_{token_type}_image.png"
        original_patch_dataurl = dataurl
        original_image_dataurl = convert_image_to_base64(f"original_images/original_image_{filename_prefix}.png")
    else:
        dataurl = convert_image_to_base64(f"llava_image_patches/{token_type}_{filename_prefix}_patch_{i-1}.png")
        original_patch_dataurl = dataurl
        original_image_dataurl = "null"
    
    row = 0 if i == 0 else (i - 1) // (336 // patch_size)
    col = 0 if i == 0 else (i - 1) % (336 // patch_size)
    ad_row = row
    ad_col = -1 if i == 0 else col
    
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
    # input_ids = inputs["input_ids"][0]
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
