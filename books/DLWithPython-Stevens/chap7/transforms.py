from torchvision import transforms
from PIL import Image

def view_img_tranform(img_path, transform):
    """ 
    Return image after transformation

    Parameters
    ----------
    img_path: str
        relative/absolute path to image
    transform: torchvision.transforms
        transform to be applied on image

    Returns
    -------
    """
    # get image
    img = Image.open(img_path)
    img.show()

    # apply transform
    trans_img = transform(img)
    trans_img.show()


# check all transforms available
print(dir(transforms))

# instanciating the transform
#  transform = transforms.CenterCrop(400)
#  transform = transforms.RandomAffine(40)
#  transform = transforms.RandomSolarize(3)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.324, 0.452, 0.198], [0.194, 0.243, 0.217]),
    transforms.ToPILImage(),
])

# transforming the image
img_path = "books/DLWithPython-Stevens/chap2/doggo.jpg"
view_img_tranform(img_path, transform)



