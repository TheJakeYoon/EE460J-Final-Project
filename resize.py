import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

class Resize():

    def __init__(self, style_img, content_img):
        '''Class called with style image and content as numpy arrays''' 
        self.style_img = style_img
        self.content_img = content_img
    
    def make_tensor(self, any_img):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = transforms.Compose([
            transforms.ToTensor()])  # transform it into a torch tensor
        # fake batch dimension required to fit network's input dimensions
        image = loader(any_img).unsqueeze(0)
        return image.to(device, torch.float)
    
    def resized_tensors(self):
        '''Resizes'''
        content_img_tensor = self.make_tensor(self.content_img)
        height = content_img_tensor.shape[2]
        width = content_img_tensor.shape[3]
        style_img_resized = self.style_img.resize((width, height), Image.ANTIALIAS)
        self.style_img_tensor = self.make_tensor(style_img_resized)
        return self.style_img_tensor, content_img_tensor
    
    def plot_resized(self):
        image = self.style_img_tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension
        unloader = transforms.ToPILImage()
        image = unloader(image)
        return plt.imshow(image)
