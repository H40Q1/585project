from __future__ import print_function
import torch.optim as optim
from content_presentation import get_style_model_and_losses


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1,
                       content_layers=['conv_4'],
                       style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):

    """Run the style transfer."""
    print('Building the style transfer model..')

    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img, content_img, content_layers, style_layers)

    optimizer = optim.Adam([input_img.requires_grad_()], lr=0.1, eps=1e-1)


    print('Optimizing..')
    step_i = 0
    while step_i <= num_steps:
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_img)
        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss

        style_score *= style_weight
        content_score *= content_weight

        loss = style_score + content_score
        loss.backward()
        optimizer.step()

        # step_i += 1
        # if step_i % 50 == 0:
        #     print("run {}:".format(step_i))
        #     print('Style Loss : {:4f} Content Loss: {:4f}'.format(
        #         style_score.item(), content_score.item()))
        #     print()

        # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img
