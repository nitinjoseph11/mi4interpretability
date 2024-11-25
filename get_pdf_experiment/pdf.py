import utilities
import model_architectures
import torch
import torch.utils.data as torch_utilities
from torchvision import datasets
from torchvision import transforms
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help="Enter model architecture")
args = parser.parse_args()

tensor_transform = transforms.Compose([transforms.ToTensor()])

dataset = datasets.MNIST(root='./get_pdf_experiment/data', train=True, download=True, transform=tensor_transform)
loader = torch_utilities.DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

model_path = input("Enter the path to the model file (e.g., /home/nmadapa/get_pdf_experiment/models/AE_20230802_145730.pth): ")

# Extract encoded representations and labels
if args.model.lower() == 'resnet':
        model = model_architectures.ResNet18()
else:
    model_class = getattr(model_architectures, args.model)
    model = model_class()
model.load_state_dict(torch.load(model_path))

print(f"Model loaded successfully from {model_path}")

model.eval()

all_encoded = []
all_labels = []

with torch.no_grad():
    for image, labels in loader:
        image = image.reshape(-1, 28*28) if args.model.lower() == 'AE' else image
        encoded, latent = model(image)
        all_encoded.append(encoded) if args.model.lower() == 'AE' else all_encoded.append(latent)
        all_labels.append(labels)

all_encoded = torch.cat(all_encoded).numpy()
all_labels = torch.cat(all_labels).numpy()

all_encoded_clean = all_encoded[~np.isnan(all_encoded).any(axis=1)]
all_encoded_clean = all_encoded_clean[~np.isinf(all_encoded_clean).any(axis=1)]

if all_encoded_clean.size == 0:
    raise ValueError("All data points have been removed due to NaNs or infs.")

#density = utilities.getDensity(all_encoded_clean)

mi_data_path = input("Enter the path to the mutual information data (e.g., /home/nmadapa/get_pdf_experiment/models/AE_20230802_145730.pth): ")
mi_data = np.load(mi_data_path)
mi_layers_input = mi_data['mi_layers_input']
mi_layers_output = mi_data['mi_layers_output']

utilities.visualize(
    x=np.arange(len(mi_layers_input)),
    y=mi_layers_input,
    xlabel='Layer',
    ylabel='Mutual Information (Input)',
    title='Layerwise MI with Input',
    plot_type='scatter',
    filename='mi_input_plot.jpg'
)

utilities.visualize(
    x=np.arange(len(mi_layers_output)),
    y=mi_layers_output,
    xlabel='Layer',
    ylabel='Mutual Information (Output)',
    title='Layerwise MI with Output',
    plot_type='scatter',
    filename='mi_output_plot.jpg'
)


# utilities.visualize(
#     x=density,
#     xlabel='Density',
#     ylabel='Frequency',
#     title='p(u/x)',
#     plot_type='hist',
#     bins=50,    # Number of bins
#     log=True ,
#     filename='plot_1.jpg'   # Logarithmic scale for y-axis
# )