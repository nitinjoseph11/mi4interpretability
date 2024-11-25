import numpy as np

# Load the .npz file
mi_data = np.load('/home/nmadapa/get_pdf_experiment/models/ResNet_20240911_190330_single_sample_mi_progression.npz')

# Print the available keys
print("Available keys in the .npz file:")
for key in mi_data.keys():
    print(key)