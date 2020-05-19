import sys
import train_args
import torch

from torchvision import models
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def main():
    args = train_args.get_args()

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

    data_dir = args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    training_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)

    # Start with CPU
    device = torch.device("cpu")

    print("Cuda: " + str(torch.cuda.is_available()))

    # Requested GPU
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("Using CPU.")

    model = models.__dict__[args.arch](pretrained=True)

    input_size = model.classifier[0].in_features

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

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

    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in training_dataloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_dataloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        val_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Validation loss: {val_loss / len(valid_dataloader):.3f}.. "
                      f"Validation accuracy: {accuracy / len(valid_dataloader):.3f}")
                running_loss = 0
                model.train()

    print('Saving')

    model.class_to_idx = training_dataset.class_to_idx
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

