import torch

from torchvision import models, transforms
from PIL import Image

# view all available pretrained models
#  print(dir(models))

# import pretrained models
alexnet = models.AlexNet()
resnet = models.resnet101(pretrained=True)

# preprocess image to tensor
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.486, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
    ])


# print image before and after preprocessing
img = Image.open('books/DLWithPython-Stevens/chap2/doggo.jpg')
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)
out_t = batch_t.data.squeeze()
out_img = transforms.ToPILImage()(out_t)
out_img.show()
img.show()

#  resnet.eval()
#  out = resnet(batch_t)
#  print(out)



