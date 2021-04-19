from __future__ import print_function
import torch
from PIL import Image
import torchvision.transforms as transforms


def loader(style, content, device):

    # desired size of the output image
    imsize = (128, 128)

    # scale imported image + transform it into a torch tensor
    loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

    def image_loader(image_name):
        image = Image.open(image_name).convert('RGB')
        # fake batch dimension required to fit network's input dimensions
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)

    style_img = image_loader(style)
    content_img = image_loader(content)

    assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

    return style_img, content_img