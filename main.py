from get_img import GetImage
from clippa import Clippy
from resize import Resize
from PIL import Image
import torch
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pick any image hosted on the internet or locally
#path = 'https://www.ece.utexas.edu/sites/default/files/portraits/Dimakis-Alex.png'
path = './temp.jpg'

# pass first arg 'online' or 'offline'
content = GetImage('oasdnline', path)
content_img_open = content.img_open()
plot1 = content.show(content_img_open, 'Content image of choice')

# input clip string label
label = "man wearing glasses"
clip = Clippy()
df_clip = clip.process(label)
plot2 = clip.plot()
style_img_open = Image.open('./unsplash/' + df_clip.iloc[0,:].name)
style_img_open = Image.open('./unsplash/2r5adxul49E.jpg')
plot2 = content.show(style_img_open, 'Style image of choice')

# resize the chosen style_image to content_image size
resize = Resize(style_img_open, content_img_open)
style_img_resized, content_img_resized = resize.resized_tensors()

assert style_img_resized.shape == content_img_resized.shape, \
        "Dimension mismatch: style and content images are of different size"

plot3 = resize.plot_resized()