from PIL import Image
import urllib.request
import matplotlib.pyplot as plt

class GetImage():

    def __init__(self, mode, path):
        self.mode = mode
        self.path = path
        self.img = None

    def img_open(self):
        if self.mode == 'online':
            with urllib.request.urlopen(self.path) as url:
                with open('temp.jpg', 'wb') as f:
                    f.write(url.read())
            self.img = Image.open('temp.jpg')
        else:
            self.img = Image.open(self.path)
        return self.img.convert("RGB")

    def show(self, any_img, title):
        plt.imshow(any_img)
        plt.axis('off')
        plt.title(title)
        return plt.show()