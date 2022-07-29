import torch 

from PIL import Image
from torchvision import transforms


def generate_random_img(height, width, is_colored=True):
    """ 
    Generate random colored image of size (widthxheight)
    """
    num_channels = 3 if is_colored else 1
    t_rand = torch.rand((num_channels, height, width)) 
    to_img = transforms.ToPILImage()
    out_img = to_img(t_rand)
    out_img.show()
    

generate_random_img(400, 800)
