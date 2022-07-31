from PIL import Image

from torchvision import transforms

def normalize_img(img_path):
    img = Image.open(img_path)
    preprocessing = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = preprocessing(img)
    img_tensor /= 255
    print(img_tensor)
    return transforms.ToPILImage()(img_tensor)



def main():
    img_path = 'books/DLWithPython-Stevens/chap2/doggo.jpg'
    image_normalized = normalize_img(img_path)
    image_normalized.show()
    


if __name__ == "__main__":
    main()


