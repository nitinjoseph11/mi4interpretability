import numpy as np
import matplotlib.pyplot as plt
import argparse
import model_architectures
import torch
import torch.utils.data as torch_utilities
from torchvision import datasets
from torchvision import transforms

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help="Enter model architecture")
parser.add_argument('--mi_data', type=str, required=True, help="Enter the path to mutual information data")
args = parser.parse_args()

# Prepare the CIFAR-100 dataset loader
cifar_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 mean and std
])

dataset = datasets.CIFAR100(root='./get_pdf_experiment/data', train=False, download=True, transform=cifar_transforms)
loader = torch_utilities.DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

# Load the model
model_path = input("Enter the path to the model file (e.g., /home/nmadapa/get_pdf_experiment/models/ResNet_20240911_190330.pth): ")
if args.model.lower() == 'resnet':
    model = model_architectures.ResNet18()
else:
    model_class = getattr(model_architectures, args.model)
    model = model_class()

model.load_state_dict(torch.load(model_path))
model.eval()
print(f"Model loaded successfully from {model_path}")

# Load Mutual Information Data (with allow_pickle=True)
mi_data = np.load(args.mi_data, allow_pickle=True)
mi_layers_input = mi_data['mi_layers_input_progression'].item()
mi_layers_output = mi_data['mi_layers_output_progression'].item()

# Get layers and MI for the single epoch
layers = list(mi_layers_input.keys())
mi_input = [mi_layers_input[layer][-1] for layer in layers]  # MI for inputs at the last epoch
mi_output = [mi_layers_output[layer][-1] for layer in layers]  # MI for outputs at the last epoch

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Plot MI for input
ax.plot(layers, mi_input, label='Input MI', marker='o', linestyle='-', color='blue')

# Plot MI for output
ax.plot(layers, mi_output, label='Output MI', marker='x', linestyle='--', color='orange')

# Labeling and layout
ax.set_xlabel('Layers')
ax.set_ylabel('Mutual Information')
ax.set_title('Mutual Information for a Single Sample Across Layers')
ax.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('plot_single_sample.png')
print('2D Plot saved as plot_single_sample.png')
