import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from sklearn.neighbors import KernelDensity
import datetime

from npeet import entropy_estimators as ee

def generate_model_path(model_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"/home/nmadapa/get_pdf_experiment/models/{model_name}_{timestamp}.pth"

def getDensity(clean_encoded_data):
    print('Calculating density...')
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(clean_encoded_data)
    log_density = kde.score_samples(clean_encoded_data)
    density = np.exp(log_density)  
    print('Returning density...')
    return density

#get mutual information from entropy


def visualize(x=None, y=None, 
              xlabel='', ylabel='', title='', 
              plot_type='scatter', filename='plot.jpg', **kwargs):
    plt.figure(figsize=(8, 6))
    
    if plot_type == 'scatter':
        if x is not None and y is not None:
            scatter = plt.scatter(x, y, **kwargs)
            plt.colorbar(scatter)
    elif plot_type == 'hist':
        if x is not None:
            plt.hist(x, **kwargs)
    else:
        raise ValueError("Unsupported plot_type. Use 'scatter' or 'hist'.")
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename, format='jpg')
    plt.close()
