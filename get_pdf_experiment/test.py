import model_architectures
import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
from npeet import entropy_estimators as ee
import argparse
import random
import utilities

def pad_tensors(x, y):
    max_len = max(x.size(0), y.size(0))
    x_padded = torch.nn.functional.pad(x, (0, max_len - x.size(0)))
    y_padded = torch.nn.functional.pad(y, (0, max_len - y.size(0)))
    return x_padded, y_padded

def test_random_sample(model, loader, input_transform_fn=lambda x: x):
    
    model.eval()  # Set the model to evaluation mode
    
    # Select a random batch from the loader
    sample_data, sample_label = next(iter(loader))
    random_idx = random.randint(0, sample_data.size(0) - 1)  # Select a random index in the batch
    random_sample_data = sample_data[random_idx:random_idx+1].to(device)  # Random sample
    random_sample_label = sample_label[random_idx:random_idx+1].to(device)

    # Run a dummy forward pass to initialize the activation keys
    model(random_sample_data)

    # Initialize MI progression dictionaries for the random tracked sample
    mi_layers_input_progression = {key: [] for key in model.activations.keys()}
    mi_layers_output_progression = {key: [] for key in model.activations.keys()}

    with torch.no_grad():  # No need for gradient calculation during evaluation
        # Perform a forward pass to get activations and final output for the random sample
        final_output_sample, _ = model(random_sample_data)
        
        for key, activations in model.activations.items():
            # Compute MI for input to activations for this random sample
            sample_padded, activation_padded = pad_tensors(random_sample_data.flatten(), activations.flatten())
            mi_input = ee.midd(sample_padded.cpu().detach().numpy(), activation_padded.cpu().detach().numpy())
            mi_layers_input_progression[key].append(mi_input)

            # Compute MI for output to activations for this random sample
            final_output_padded, activation_padded = pad_tensors(final_output_sample.flatten(), activations.flatten())
            mi_output = ee.midd(final_output_padded.cpu().detach().numpy(), activation_padded.cpu().detach().numpy())
            mi_layers_output_progression[key].append(mi_output)

    # Save the MI progression data for the random sample
    mi_data_path = utilities.generate_model_path(model.__class__.__name__).replace('.pth', '_random_sample_mi_progression.npz')
    np.savez(mi_data_path, mi_layers_input_progression=mi_layers_input_progression, mi_layers_output_progression=mi_layers_output_progression)
    print(f"Random sample MI progression data saved at: {mi_data_path}")

    return mi_data_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Enter model architecture")
    parser.add_argument('--model_path', type=str, required=True, help="Enter trained model path")
    
    args = parser.parse_args()

    # Load the model architecture
    if args.model.lower() == 'resnet':
        model = model_architectures.ResNet18()
    else:
        model_class = getattr(model_architectures, args.model)
        model = model_class()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f'Using device: {device}')

    # Load the pre-trained model
    model.load_state_dict(torch.load(args.model_path))
    print(f"Loaded model from {args.model_path}")

    # Define the test data transformations
    cifar_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 mean and std
    ])

    # Load the CIFAR-100 test dataset
    test_dataset = datasets.CIFAR100(root='./get_pdf_experiment/data', train=False, download=True, transform=cifar_transforms)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # If model is not ResNet, reshape the input appropriately
    input_transform_fn = lambda x: x if args.model.lower() == 'resnet' else x.view(x.size(0), -1)

    # Evaluate the model on the test set and track MI progression for a random sample
    mi_data_path = test_random_sample(model, test_loader, input_transform_fn)
    print(f"Mutual Information progression data for random sample saved at: {mi_data_path}")
