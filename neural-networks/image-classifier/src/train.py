import sys
import train_args
import torch

from torchvision import models
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def main():
    args = train_args.get_args()
    print(args.data_directory)
    print(args.save_dir)

    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_size = 224
    batch_size = 32

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(image_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, std)])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, std)])

    training_dataset = datasets.ImageFolder(args.data_directory, transform=train_transforms)

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    # Start with CPU
    device = torch.device("cpu")

    print("Cuda: " + str(torch.cuda.is_available()))

    # Requested GPU
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("Using CPU.")

    # TODO [VK]: Select the training model based on input
    model = models.__dict__[args.arch](pretrained=True)

    input_size = model.classifier[0].in_features

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # TODO [VK] These should become dynamic
    model_classifier = nn.Sequential(nn.Linear(input_size, args.hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(args.hidden_units, 102),
                                     nn.LogSoftmax(dim=1))

    model.classifier = model_classifier

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    model.to(device)

    # TODO The training loss, validation loss, and validation accuracy are printed out as a network trains

    print_every = 5
    for epoch in range(args.epochs):
        epoch_loss = 0
        prev_chk = 0
        total = 0
        correct = 0
        print(f'\nEpoch {epoch + 1} of {args.epochs}\n----------------------------')
        for batch_index, (images, labels) in enumerate(training_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Keep a running total of loss for
            # this epoch
            epoch_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Keep a running total of loss for
            # this epoch
            itr = (batch_index + 1)
            if itr % print_every == 0:
                avg_loss = f'avg. loss: {epoch_loss / itr:.4f}'
                acc = f'accuracy: {(correct / total) * 100:.2f}%'
                print(f'  Batches {prev_chk:03} to {itr:03}: {avg_loss}, {acc}.')
                prev_chk = (batch_index + 1)

    print('Saving')

    checkpoint = {'input_size': input_size,
                  'output_size': 102,
                  'arch': args.arch,
                  'learning_rate': args.learning_rate,
                  'batch_size': 64,
                  'classifier': model_classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, f'{args.save_dir}/checkpoint_cli.pth')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

