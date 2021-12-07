from get_img import GetImage
from clippa import Clippy
from resize import Resize
from transfer import CyberpunkStyle
from metrics import Metrics
from PIL import Image
import numpy as np
import torch
import torchvision.utils as utils
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning) 
# pick any image hosted on the internet or locally
path = 'https://www.ece.utexas.edu/sites/default/files/portraits/Dimakis-Alex.png'
#path = './temp.jpg'

# pass first arg 'online' or 'offline'
content = GetImage('online', path)
content_img_open = content.img_open()
plot1 = content.show(content_img_open, 'Content image of choice')

# input clip string label (highlighted item to cyberpunkify), ~3min to complete
label = "man wearing glasses"
data_path = './unsplash/'
clip = Clippy(label, data_path)
df_clip = clip.process()
plot2 = clip.plot()
style_img_open = Image.open(data_path + df_clip.iloc[3,:].name)

#style_img_open = Image.open('./unsplash/470eBDOc8bk.jpg')
plot2 = content.show(style_img_open, 'Style image of choice')

# resize the chosen style_image to content_image size
resize = Resize(style_img_open, content_img_open)
style_img_resized, content_img_resized = resize.resized_tensors()
# input vs style metrics
# metrics = Metrics(style_img_resized, content_img_resized)
# distance = metrics.mse()
# similarity = metrics.ssim()
# print(f'MSE:{distance}, SSIM:{similarity}')
# print(distance*similarity)

assert style_img_resized.shape == content_img_resized.shape, \
        "Dimension mismatch: style and content images are of different size"

plot3 = resize.plot_resized()

# tolerance == early stopping criteria (absolute difference in style_loss)
transfer = CyberpunkStyle(style_img_resized, content_img_resized, 
                            num_steps=900, style_weight=1e6, tolerance=0.0001)

cyberpunk_img, style_loss_list, content_loss_list = transfer.run()
show_output = transfer.show_img()

plot5 = transfer.loss_plot()

output_metrics = Metrics(cyberpunk_img, content_img_resized)
output_distance = output_metrics.mse()
output_similarity = output_metrics.ssim()
print(f'Output\nMSE:{output_distance}, SSIM:{output_similarity}')

# utils.save_image(cyberpunk_img, 'cyberpunk.png')
