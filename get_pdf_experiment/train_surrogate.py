import model_architectures
import torch
from torchvision import datasets
from torchvision import transforms
import utilities
import argparse
from npeet import entropy_estimators as ee
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR

def pad_tensors(x, y):
    max_len = max(x.size(0), y.size(0))
    x_padded = torch.nn.functional.pad(x, (0, max_len - x.size(0)))
    y_padded = torch.nn.functional.pad(y, (0, max_len - y.size(0)))
    return x_padded, y_padded

def train_model(model, epochs, loader, loss_function, optimizer, input_transform_fn=lambda x: x):
    
    model.train()
    
    sample_data, sample_label = next(iter(loader))
    sample_data = sample_data[0:1].to(device) 
    sample_label = sample_label[0:1].to(device)

    swa_model = AveragedModel(model)
    model(sample_data)
    scheduler = SWALR(optimizer, anneal_strategy="cos", anneal_epochs=5, swa_lr=1e-3)

    mi_layers_input_progression = {key: [] for key in model.activations.keys()}
    mi_layers_output_progression = {key: [] for key in model.activations.keys()}

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:
            images = input_transform_fn(images).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            final_output, latent_features = model(images)
            loss = loss_function(final_output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        swa_model.update_parameters(model)

        with torch.no_grad():
            final_output_sample, _ = model(sample_data)

            for key, activations in model.activations.items():
                # Compute MI for input to activations
                sample_padded, activation_padded = pad_tensors(sample_data.flatten(), activations.flatten())
                mi_input = ee.midd(sample_padded.cpu().detach().numpy(), activation_padded.cpu().detach().numpy())
                mi_layers_input_progression[key].append(mi_input)

                # Compute MI for output to activations
                final_output_padded, activation_padded = pad_tensors(final_output_sample.flatten(), activations.flatten())
                mi_output = ee.midd(final_output_padded.cpu().detach().numpy(), activation_padded.cpu().detach().numpy())
                mi_layers_output_progression[key].append(mi_output)

        scheduler.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}')
        print(f'MI progression after epoch {epoch+1}:')
        for key in mi_layers_input_progression.keys():
            print(f'{key} - Input MI: {mi_layers_input_progression[key][-1]}, Output MI: {mi_layers_output_progression[key][-1]}')

    model_path = utilities.generate_model_path(model.__class__.__name__)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    mi_data_path = model_path.replace('.pth', '_single_sample_mi_progression.npz')
    np.savez(mi_data_path, mi_layers_input_progression=mi_layers_input_progression, mi_layers_output_progression=mi_layers_output_progression)
    print(f"MI data for single sample saved to {mi_data_path}")

    return model_path, mi_data_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Enter model architecture")
    parser.add_argument('--epochs', type=int, required=True, help="Enter number of epochs")
    
    args = parser.parse_args()

    if args.model.lower() == 'resnet':
        model = model_architectures.ResNet18()
    else:
        model_class = getattr(model_architectures, args.model)
        model = model_class()

    epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  
    print(f'Using device: {device}')

    #mnist

    # tensor_transform = transforms.ToTensor()
    # dataset = datasets.MNIST(root='./get_pdf_experiment/data', train=True, download=True, transform=tensor_transform)
    # loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

    #cifar100 

    # cifar_transforms = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 mean and std
    # ])

    # dataset = datasets.CIFAR100(root='./get_pdf_experiment/data', train=True, download=True, transform=cifar_transforms)
    # loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

    # Tiny ImageNet

    # tiny_imagenet_transforms = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(64, padding=4),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Tiny ImageNet mean and std
    # ])

    # dataset = datasets.ImageFolder(root='./tiny-imagenet-200/train', transform=tiny_imagenet_transforms)
    # loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

    # pet_transforms = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomResizedCrop(224),  # Resize to 224x224 for standard architectures
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Mean and std for pre-trained models
    # ])

    # dataset = datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=pet_transforms)
    # loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

    cats_and_dogs_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 for compatibility with ResNet
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Mean and std for pre-trained models
])

# Load the train and validation datasets
    dataset = ImageFolder(root='./data/cats_and_dogs_filtered/train', transform=cats_and_dogs_transforms)
    loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
    
    loss_function = torch.nn.CrossEntropyLoss() if args.model.lower() == 'resnet' else torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-12, weight_decay=1e-8)

    input_transform_fn = lambda x: x if args.model.lower() == 'resnet' else x.view(x.size(0), -1)

    model_path, mi_data_path = train_model(model, epochs, loader, loss_function, optimizer, input_transform_fn)
    print(f"Mutual Information data saved at: {mi_data_path}")
    print(f"Trained model saved at: {model_path}")
