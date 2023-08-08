import streamlit as st
import os 
import numpy as np

from tqdm import tqdm
from PIL import Image

import torch
import clip


#Path


IMAGE_KEYFRAME_PATH = "D:/Workspace/Project/AIC2023/AIC2022/HCMAI22_MiniBatch1/HCMAI22_MiniBatch1/Keyframes"
VISUAL_FEATURES_PATH = "D:/Workspace/Project/AIC2023/AIC2022/HCMAI22_MiniBatch1/HCMAI22_MiniBatch1/CLIP_features"

class TextEmbedding():
  def __init__(self):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model, _ = clip.load("ViT-B/16", device=self.device)

  def __call__(self, text: str) -> np.ndarray:
    text_inputs = clip.tokenize([text]).to(self.device)
    with torch.no_grad():
        text_feature = self.model.encode_text(text_inputs)[0]
    
    return text_feature.detach().cpu().numpy()
  

# ==================================
text = "A car is parked on the road"
text_embedd = TextEmbedding()
text_feat_arr = text_embedd(text)
print(text_feat_arr.shape, type(text_feat_arr))



from typing import List, Tuple
def indexing_methods() -> List[Tuple[str, int, np.ndarray],]:
  db = []
 
  for feat_npy in tqdm(os.listdir(VISUAL_FEATURES_PATH)):
    video_name = feat_npy.split('.')[0]
    feats_arr = np.load(os.path.join(VISUAL_FEATURES_PATH, feat_npy))
    for idx, feat in enumerate(feats_arr):
 
      instance = (video_name, idx, feat)
      db.append(instance)
  return db


# ==================================
visual_features_db = indexing_methods()
print()
print(visual_features_db[0][:2], visual_features_db[0][-1].shape)

def search_engine(query_arr: np.array, 
                  db: list, 
                  topk:int=10, 
                  measure_method: str="dot_product") -> List[dict,]:
  
 
  measure = []
  for ins_id, instance in enumerate(db):
    video_name, idx, feat_arr = instance

    if measure_method=="dot_product":
      distance = query_arr @ feat_arr.T
    elif measure_method=="l1_norm":
      distance = -1 * np.mean([abs(q - t) for q, t in zip(query_arr, feat_arr)])
    measure.append((ins_id, distance))
  

  measure = sorted(measure, key=lambda x:x[-1], reverse=True)
  

  search_result = []
  for instance in measure[:topk]:
    ins_id, distance = instance
    video_name, idx, _ = db[ins_id]

    search_result.append({"video_name":video_name,
                          "keyframe_id": idx,
                          "score": distance})
  return search_result


# ==================================
search_result = search_engine(text_feat_arr, visual_features_db, 10)
print(search_result)


def read_image(results: List[dict,]) -> List[Image.Image,]:
  images = []
  for res in results:
    image_file = sorted(os.listdir(os.path.join(IMAGE_KEYFRAME_PATH, res["video_name"])))[res["keyframe_id"]]
    image_path = os.path.join(IMAGE_KEYFRAME_PATH, res["video_name"], image_file)
    image = Image.open(image_path)
    images.append(image)
  return images

def visualize(imgs: List[Image.Image, ]) -> None:
    rows = len(imgs) // 5
    if not rows:
      rows += 1
    cols = len(imgs) // rows
    if rows*cols < len(imgs):
      rows += 1
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    display(grid)


# ==================================
images = read_image(search_result)

st.write("""
# AIC 2022 WEB

""")

st.sidebar.header('Chỗ để nhập querry')


def user_input():
    text = st.sidebar.text_input('Nhập querry vào đây')

    return text
input = user_input()

topk = st.slider('Topk', 0,50)
measure_method = st.select_slider('measure_method:', ['dot_product', 'l1_norm'])


text_feat_arr = text_embedd(input)
search_result = search_engine(text_feat_arr, visual_features_db, int(topk), measure_method)
images = read_image(search_result)




st.subheader('Dự đoán')
if input == '':
    st.write('Kết quả dựa trên truy vấn:')
else:
    for im in images:
        st.image(im, caption="Ảnh truy vấn ", use_column_width=True)  
    st.write(search_result)
    
