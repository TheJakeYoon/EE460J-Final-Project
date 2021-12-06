from skimage.metrics import structural_similarity as ssim
import numpy as np

class Metrics():

    def __init__(self, style_img, content_img):
        self.style_img = style_img[0][0].cpu().detach().numpy()
        self.content_img = content_img[0][0].cpu().detach().numpy()
        self.cyber_ssim = None
        self.cyber_mse = None
    
    def ssim(self):
        self.cyber_ssim = ssim(self.style_img, self.content_img,
                  data_range=self.style_img.max() - self.style_img.min())
        return self.cyber_ssim

    def mse(self):
        self.cyber_mse = np.linalg.norm(self.style_img - self.content_img)
        return self.cyber_mse