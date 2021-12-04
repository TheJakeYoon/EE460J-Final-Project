from datetime import datetime
import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

class Clippy():

    def __init__(self, label_str):
        self.label = label_str # "man wearing glasses"
        self.mypath = './unsplash/'
        self.df_res = pd.DataFrame()

    def process(self):
        device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        label_list = [self.label, ""]
        text = clip.tokenize(label_list).to(device)

        img_names = [f for f in listdir(self.mypath) if isfile(join(self.mypath, f))]
        #labl_dict = {label: None for label in label_list}
        img_labl_dict = {}
        img_tensor_dict = {img_name: preprocess(Image.open(self.mypath + f'/{img_name}')).unsqueeze(0).to(device) for img_name in img_names}

        start_time = datetime.now()
        print('Labeling intiated...')

        for img_name in img_names:
            image = img_tensor_dict[img_name]
            probs = []
            logits_per_image, logits_per_text = model(image, text)
            with torch.no_grad():
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            img_labl_dict[img_name] = {label: probs[0][ix] for ix, label in enumerate(label_list)}

        end_time = datetime.now()
        print(f'Labeling completed in: {str(end_time-start_time)}')
        self.df_res = pd.DataFrame.from_dict(img_labl_dict).T.sort_values(by=label_list[0], ascending=False)
        return self.df_res.iloc[:5,:]
    
    def plot(self):
        fig, ax = plt.subplots(1,5, figsize=(12,30))
        for i in range(5):
            ax[i].imshow(Image.open(self.mypath+self.df_res.iloc[i,:].name))
            ax[i].axis('off')
            ax[i].title.set_text(f'{i} \n {np.round(self.df_res.iloc[i,0], 4)} \n {self.df_res.iloc[i,:].name}')
        return plt.show()
