# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def plot_mi_progression(mi_input, mi_output, epochs):
#     """
#     Plots the MI progression over epochs for each layer using 3D plot.
    
#     :param mi_input: Dictionary containing MI values for input to layers across epochs
#     :param mi_output: Dictionary containing MI values for output from layers across epochs
#     :param epochs: Total number of epochs
#     """
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Get layer names from the dictionary keys
#     layers = list(mi_input.keys())
#     num_layers = len(layers)

#     # Create mesh grids for plotting
#     epoch_grid, layer_grid = np.meshgrid(range(epochs), range(num_layers))

#     # Plot MI for inputs
#     mi_input_values = np.array([mi_input[layer] for layer in layers])
#     ax.plot_surface(epoch_grid, layer_grid, mi_input_values, cmap='viridis', alpha=0.7, label='Input MI')

#     # Plot MI for outputs
#     mi_output_values = np.array([mi_output[layer] for layer in layers])
#     ax.plot_surface(epoch_grid, layer_grid, mi_output_values, cmap='plasma', alpha=0.7, label='Output MI')

#     # Labels and title
#     ax.set_xlabel('Epochs')
#     ax.set_ylabel('Layers')
#     ax.set_zlabel('MI Value')
#     ax.set_title('Mutual Information Progression Over Epochs')
#     ax.set_yticks(range(num_layers))
#     ax.set_yticklabels(layers)
    
#     plt.savefig('output_training_progression.png')

# # Load the MI data
# mi_data = np.load('/home/nmadapa/get_pdf_experiment/models/ResNet_20241023_164739_single_sample_mi_progression.npz', allow_pickle=True)
# mi_input_progression = mi_data['mi_layers_input_progression'].item()
# mi_output_progression = mi_data['mi_layers_output_progression'].item()

# # Visualize the MI progression
# plot_mi_progression(mi_input_progression, mi_output_progression, epochs=250)

import numpy as np
import matplotlib.pyplot as plt

def plot_mutual_information(mi_data_path):
    data = np.load(mi_data_path, allow_pickle=True)
    mi_layers_input_progression = data['mi_layers_input_progression'].item()
    mi_layers_output_progression = data['mi_layers_output_progression'].item()

    plt.figure(figsize=(10, 8))

    # Loop over each layer to plot the MI(Input, Latent) vs MI(Latent, Output) for all epochs
    for layer_name, input_mi_values in mi_layers_input_progression.items():
        output_mi_values = mi_layers_output_progression[layer_name]
        
        # Scatter plot of MI(Input, Latent) vs MI(Latent, Output) for each epoch
        plt.scatter(input_mi_values, output_mi_values, label=f'Layer {layer_name}', alpha=0.7)
    
    # Labeling the plot
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('MI(Input, Latent)')
    plt.ylabel('MI(Latent, Output)')
    plt.title('Mutual Information (Input vs Output) Across Layers and Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('mi_input_vs_output_2d_multiple_layers.jpg')
    plt.show()

# Path to the mutual information data
mi_data_path = '/home/nmadapa/get_pdf_experiment/models/ResNet_20241114_222002_single_sample_mi_progression.npz'  # Replace with your actual path
plot_mutual_information(mi_data_path)