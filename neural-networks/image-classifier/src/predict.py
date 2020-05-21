import sys
import json
import torch
import predict_args
import numpy as np

from PIL import Image
from torchvision import models


def main():
    args = predict_args.get_args()

    device = torch.device("cpu")

    if args.use_gpu:
        device = torch.device("cuda:0")

    # load categories
    with open(args.categories_json, 'r') as f:
        cat_to_name = json.load(f)

    # load model
    model = load_checkpoint(device, args.checkpoint_file)

    top_prob, top_classes = predict(device, args.path_to_image, model, args.top_k)

    label = top_classes[0]
    prob = top_prob[0]

    print(f'Parameters\n---------------------------------')

    print(f'Image  : {args.path_to_image}')
    print(f'Model  : {args.checkpoint_file}')
    print(f'Device : {device}')

    print(f'\nPrediction\n---------------------------------')

    print(f'Flower      : {cat_to_name[label]}')
    print(f'Label       : {label}')
    print(f'Probability : {prob*100:.2f}%')

    print(f'\nTop K\n---------------------------------')

    for i in range(len(top_prob)):
        print(f"{cat_to_name[top_classes[i]]:<25} {top_prob[i]*100:.2f}%")


def load_checkpoint(device, file='checkpoint.pth'):
    checkpoint = torch.load(file)

    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.to(device)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image):
    size = 256, 256
    image.thumbnail(size)

    width, height = image.size  # Get dimensions
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    image = image.crop((left, top, right, bottom))

    np_image = np.array(image)

    np_image = np_image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    new_np_image = (np_image - mean) / std

    transposed = new_np_image.transpose((2, 0, 1))

    to_float_tensor = torch.from_numpy(transposed).type(torch.FloatTensor)

    return to_float_tensor


def predict(device, image_path, model, topk=5):
    flower = Image.open(image_path)
    with torch.no_grad():
        model.eval()

        flower_input = process_image(flower)
        flower_input = flower_input.to(device)
        flower_input = flower_input.unsqueeze_(0)

        output = model.forward(flower_input)
        top_prob, top_labels = torch.topk(output, topk)
        top_prob = top_prob.exp()

        idx_to_class = {y: x for x, y in model.class_to_idx.items()}

        mapped_classes = list()

        top_labels = top_labels.cpu()
        top_prob = top_prob.cpu()

        for label in top_labels.numpy()[0]:
            mapped_classes.append(idx_to_class[label])

        return top_prob.numpy()[0], mapped_classes


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
