# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# import utilities
# import model_architectures
# import torch
# import torch.utils.data as torch_utilities
# from torchvision import datasets
# from torchvision import transforms

# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, required=True, help="Enter model architecture")
# parser.add_argument('--mi_data', type=str, required=True, help="Enter the path to mutual information data")
# args = parser.parse_args()

# tensor_transform = transforms.Compose([transforms.ToTensor()])

# dataset = datasets.MNIST(root='./get_pdf_experiment/data', train=True, download=True, transform=tensor_transform)
# loader = torch_utilities.DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

# model_path = input("Enter the path to the model file (e.g., /home/nmadapa/get_pdf_experiment/models/ResNet_20240911_190330.pth): ")

# # Load the model
# if args.model.lower() == 'resnet':
#     model = model_architectures.ResNet18()
# else:
#     model_class = getattr(model_architectures, args.model)
#     model = model_class()
# model.load_state_dict(torch.load(model_path))
# model.eval()

# print(f"Model loaded successfully from {model_path}")

# # Load Mutual Information Data
# mi_data = np.load(args.mi_data, allow_pickle=True)  # Set allow_pickle=True
# mi_layers_input = mi_data['mi_layers_input_progression'].item()
# mi_layers_output = mi_data['mi_layers_output_progression'].item()

# # Plot MI progression
# epochs = len(next(iter(mi_layers_input.values())))

# fig, ax = plt.subplots(figsize=(14, 8))

# for layer in mi_layers_input.keys():
#     # Extract MI values
#     mi_input = mi_layers_input[layer]
#     mi_output = mi_layers_output[layer]
    
#     # Plot MI for input
#     ax.plot(range(epochs), mi_input, label=f'{layer} - Input MI', linestyle='-', marker='o')
    
#     # Plot MI for output
#     ax.plot(range(epochs), mi_output, label=f'{layer} - Output MI', linestyle='--', marker='x')

# ax.set_xlabel('Epoch')
# ax.set_ylabel('Mutual Information')
# ax.set_title('Mutual Information Progression Across Epochs')
# ax.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('plot_1.png')
# print('Plot saved as plot_1.png')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import utilities
import model_architectures
import torch
import torch.utils.data as torch_utilities
from torchvision import datasets
from torchvision import transforms

# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, required=True, help="Enter model architecture")
# parser.add_argument('--mi_data', type=str, required=True, help="Enter the path to mutual information data")
# args = parser.parse_args()

# tensor_transform = transforms.Compose([transforms.ToTensor()])

# dataset = datasets.MNIST(root='./get_pdf_experiment/data', train=True, download=True, transform=tensor_transform)
# loader = torch_utilities.DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

# model_path = input("Enter the path to the model file (e.g., /home/nmadapa/get_pdf_experiment/models/ResNet_20240911_190330.pth): ")

# # Load the model
# if args.model.lower() == 'resnet':
#     model = model_architectures.ResNet18()
# else:
#     model_class = getattr(model_architectures, args.model)
#     model = model_class()
# model.load_state_dict(torch.load(model_path))
# model.eval()

# print(f"Model loaded successfully from {model_path}")

# # Load Mutual Information Data
# mi_data = np.load(args.mi_data, allow_pickle=True)
# mi_layers_input = mi_data['mi_layers_input_progression'].item()
# mi_layers_output = mi_data['mi_layers_output_progression'].item()

# # Prepare data for 3D plot
# epochs = len(next(iter(mi_layers_input.values())))
# layers = list(mi_layers_input.keys())
# num_layers = len(layers)

# # Create arrays for plotting
# X, Y = np.meshgrid(range(epochs), range(num_layers))
# Z_input = np.zeros((num_layers, epochs))
# Z_output = np.zeros((num_layers, epochs))

# for i, layer in enumerate(layers):
#     Z_input[i, :] = mi_layers_input[layer]
#     Z_output[i, :] = mi_layers_output[layer]

# # Plot 3D MI for Input
# fig = plt.figure(figsize=(14, 8))
# ax = fig.add_subplot(121, projection='3d')
# ax.plot_surface(X, Y, Z_input, cmap='viridis')
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Layer')
# ax.set_zlabel('Mutual Information')
# ax.set_title('Mutual Information (Input) Across Layers and Epochs')

# # Plot 3D MI for Output
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.plot_surface(X, Y, Z_output, cmap='plasma')
# ax2.set_xlabel('Epoch')
# ax2.set_ylabel('Layer')
# ax2.set_zlabel('Mutual Information')
# ax2.set_title('Mutual Information (Output) Across Layers and Epochs')

# plt.tight_layout()
# plt.savefig('plot_2.png')
# print('3D Plot saved as plot_2.png')

# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# import model_architectures
# import torch
# import torch.utils.data as torch_utilities
# from torchvision import datasets
# from torchvision import transforms

# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, required=True, help="Enter model architecture")
# parser.add_argument('--mi_data', type=str, required=True, help="Enter the path to mutual information data")
# args = parser.parse_args()

# tensor_transform = transforms.Compose([transforms.ToTensor()])

# # Prepare dataset loader
# dataset = datasets.MNIST(root='./get_pdf_experiment/data', train=True, download=True, transform=tensor_transform)
# loader = torch_utilities.DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

# # Load model
# model_path = input("Enter the path to the model file (e.g., /home/nmadapa/get_pdf_experiment/models/ResNet_20240911_190330.pth): ")
# if args.model.lower() == 'resnet':
#     model = model_architectures.ResNet18()
# else:
#     model_class = getattr(model_architectures, args.model)
#     model = model_class()

# model.load_state_dict(torch.load(model_path))
# model.eval()
# print(f"Model loaded successfully from {model_path}")

# # Load MI data with allow_pickle=True
# mi_data = np.load(args.mi_data, allow_pickle=True)
# mi_layers_input = mi_data['mi_layers_input_progression'].item()
# mi_layers_output = mi_data['mi_layers_output_progression'].item()

# # Get number of epochs
# epochs = len(next(iter(mi_layers_input.values())))

# # Plotting MI progression
# fig, ax = plt.subplots(figsize=(14, 8))

# for layer in mi_layers_input.keys():
#     # Get MI values for the input and output
#     mi_input = mi_layers_input[layer]  # MI values for input at each epoch
#     mi_output = mi_layers_output[layer]  # MI values for output at each epoch

#     # Plot MI for input with solid line
#     ax.plot(range(epochs), mi_input, label=f'{layer} - Input MI', linestyle='-', marker='o')
    
#     # Plot MI for output with dashed line
#     ax.plot(range(epochs), mi_output, label=f'{layer} - Output MI', linestyle='--', marker='x')

# # Configure plot
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Mutual Information')
# ax.set_title('Mutual Information Progression for Single Sample Across Layers and Epochs')
# ax.legend()
# plt.grid(True)
# plt.tight_layout()

# # Save the plot as 'plot_1.png'
# plt.savefig('plot_3.png')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
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

# tensor_transform = transforms.Compose([transforms.ToTensor()])
cifar_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 mean and std
    ])
# Prepare dataset loader
# dataset = datasets.MNIST(root='./get_pdf_experiment/data', train=True, download=True, transform=tensor_transform)
# loader = torch_utilities.DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)
dataset = datasets.CIFAR100(root='./get_pdf_experiment/data', train=True, download=True, transform=cifar_transforms)
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
# Load model
model_path = input("Enter the path to the model file (e.g., /home/nmadapa/get_pdf_experiment/models/ResNet_20240911_190330.pth): ")
if args.model.lower() == 'resnet':
    model = model_architectures.ResNet18()
else:
    model_class = getattr(model_architectures, args.model)
    model = model_class()

model.load_state_dict(torch.load(model_path))
model.eval()
print(f"Model loaded successfully from {model_path}")

# Load MI data with allow_pickle=True
mi_data = np.load(args.mi_data, allow_pickle=True)
mi_layers_input = mi_data['mi_layers_input_progression'].item()
mi_layers_output = mi_data['mi_layers_output_progression'].item()

# Get number of epochs and layers
epochs = len(next(iter(mi_layers_input.values())))
layers = list(mi_layers_input.keys())

# Prepare data for plotting (MI input and output over layers and epochs)
mi_input_matrix = np.array([mi_layers_input[layer] for layer in layers])  # Input MI (shape: layers x epochs)
mi_output_matrix = np.array([mi_layers_output[layer] for layer in layers])  # Output MI (shape: layers x epochs)

# Create meshgrid for plotting
X, Y = np.meshgrid(range(epochs), range(len(layers)))  # X: epochs, Y: layers

# Plot the MI Input and Output as 3D surfaces
fig = plt.figure(figsize=(14, 10))

# Input MI plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, mi_input_matrix, cmap='viridis')
ax1.set_title('Input Mutual Information Progression')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Layers')
ax1.set_zlabel('Input MI')

# Output MI plot
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, mi_output_matrix, cmap='plasma')
ax2.set_title('Output Mutual Information Progression')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Layers')
ax2.set_zlabel('Output MI')

plt.tight_layout()
plt.savefig('plot_4.png')  # Save the plot as 'plot_1.png'



