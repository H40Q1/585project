from __future__ import print_function
import torch
import torchvision.models as models
from data_loader import loader
from run_style_transfer import run_style_transfer

from os import listdir
from os.path import isfile, join



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    content_layers_selected = ['conv_4']
    style_layers_selected = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    style_weight = 1e8

    style_path = "../Style"
    content_path = "../content"

    styleDir = [f for f in listdir(style_path) if isfile(join(style_path, f))]
    contentDir = [f for f in listdir(content_path) if isfile(join(content_path, f))]


    image_no = 1

    for content in contentDir:
        for style in styleDir:
            style_img, content_img = loader(style, content, device)
            input_img = content_img.clone().detach().requires_grad_(True)

            output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                        content_img, style_img, input_img, num_steps=300, style_weight=style_weight,
                                        content_layers=content_layers_selected,
                                        style_layers=style_layers_selected, device)


            path = '../Results/img_' + str(image_no) + '.jpg'
            output.save(path, 'JPEG')
            image_no += 1

if __name__ == "__main__":
    main()