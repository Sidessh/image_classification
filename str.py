# import streamlit as st
# import clip
# import torch
# import numpy as np
# import pandas as pd
# from PIL import Image
# import glob

# from pathlib import Path

# # Load the CLIP model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# # # Load precomputed image features (you can load these outside the Streamlit app)
# # image_features = np.load("path_to_precomputed_features.npy")
# # image_ids = pd.read_csv("path_to_image_ids.csv")['image_id'].tolist()

# features_path = Path("/Users/siddhesh/Downloads/ADM_clip/features")

# # Load all the CSV files
# csv_files = sorted(features_path.glob("*.csv"))
# # Concatenate the CSV files into a DataFrame
# image_ids_data = pd.concat([pd.read_csv(file) for file in csv_files])

# # Load all the NPY files
# npy_files = sorted(features_path.glob("*.npy"))
# # Concatenate the NPY files into a single array
# image_features = np.concatenate([np.load(file) for file in npy_files])

# # Get the list of image IDs
# image_ids = image_ids_data['image_id'].tolist()


# # Streamlit app
# st.title("Image Search App")

# # User input
# search_query = st.text_input("Enter your search query:")

# if st.button("Search"):
#     if search_query:
#         result_image_ids = search(search_query, image_features, image_ids)

#         st.markdown(f"**Top {len(result_image_ids)} Results for '{search_query}':**")

#         for image_id in result_image_ids:
#             image = Image.open(f'{images_path}/{image_id}.jpg')
#             st.image(image, caption=image_id, use_column_width=True)

# def encode_search_query(search_query):
#     with torch.no_grad():
#         text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
#         text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
#     return text_encoded

# def find_best_matches(text_features, image_features, image_ids, results_count=3):
#     similarities = (image_features @ text_features.T).squeeze(1)
#     best_image_idx = (-similarities).argsort()
#     return [image_ids[i] for i in best_image_idx[:results_count]]

# def search(search_query, image_features, image_ids, results_count=3):
#     text_features = encode_search_query(search_query)
#     return find_best_matches(text_features, image_features, image_ids, results_count)

######################

import streamlit as st
import clip
import torch
import numpy as np
import pandas as pd
from PIL import Image
import glob
from pathlib import Path

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load precomputed image features and image IDs
features_path = "/Users/siddhesh/Downloads/ADM_clip/features"
image_features = np.load(f"{features_path}/features.npy")
image_ids = pd.concat([pd.read_csv(file) for file in sorted(glob.glob(f"{features_path}/*.csv"))])['image_id'].tolist()

# Streamlit app
st.title("Image Search App")

# User input
search_query = st.text_input("Enter your search query:")

def encode_search_query(search_query):
    with torch.no_grad():
        text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    return text_encoded

# def find_best_matches(text_features, image_features, image_ids, results_count=3):
#     similarities = (image_features @ text_features.T).squeeze(1)
#     best_image_idx = (-similarities).argsort()
#     return [image_ids[i] for i in best_image_idx[:results_count]]
import torch

def find_best_matches(text_features, image_features, image_ids, results_count=3):
    text_features = text_features.to(device)  # Make sure text_features are on the same device
    image_features_tensor = torch.tensor(image_features).to(device)  # Convert image_features to a PyTorch tensor
    similarities = torch.mm(image_features_tensor, text_features.T).squeeze(1)
    best_image_idx = (-similarities).argsort()
    return [image_ids[i] for i in best_image_idx[:results_count]]


@st.cache_data()  # Use st.cache_data
def search(search_query, image_features, image_ids, results_count=3):
    text_features = encode_search_query(search_query)
    return find_best_matches(text_features, image_features, image_ids, results_count)

images_path = Path("/Users/siddhesh/Downloads/data_2/Apparel/Boys/Images/images_with_product_ids")

if st.button("Search"):
    if search_query:
        result_image_ids = search(search_query, image_features, image_ids)

        st.markdown(f"**Top {len(result_image_ids)} Results for '{search_query}':**")

        for image_id in result_image_ids:
            image = Image.open(f'{images_path}/{image_id}.jpg')
            st.image(image, caption=image_id, use_column_width=True)
                 # Display the images in one row with a specified width
            #st.image(image, width=150)

