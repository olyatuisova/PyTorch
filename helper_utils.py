import random
import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image



def display_image(image, label, title, num_ticks=6, show_values=True):
    
    vmin_val, vmax_val = None, None
    image_data = None

    if isinstance(image, Image.Image):
        vmin_val = 0
        vmax_val = 255
        image_data = np.array(image)
    elif isinstance(image, torch.Tensor):
        image_np = image.numpy().squeeze()
        vmin_val = image_np.min()
        vmax_val = image_np.max()
        image_data = image_np
    else:
        print("Warning: Unsupported image type.")
        return

    plt.figure(figsize=(9, 9))
    plt.imshow(image_data, cmap='gray', vmin=vmin_val, vmax=vmax_val)
    plt.title(f'{title} | Label: {label}')

    if show_values:
        threshold = (vmin_val + vmax_val) / 2.0
        height, width = image_data.shape
        
        for y in range(height):
            for x in range(width):
                value = image_data[y, x]
                text_color = "white" if value < threshold else "black"
                text_to_display = f"{value:.0f}" if isinstance(value, np.integer) else f"{value:.1f}"
                plt.text(x, y, text_to_display, 
                         ha="center", va="center", color=text_color, fontsize=6)

    plt.grid(True, color='red', alpha=0.3, zorder=2)
    plt.xticks(np.arange(0, 28, 4))
    plt.yticks(np.arange(0, 28, 4))
    
    cbar = plt.colorbar()
    ticks = np.linspace(vmin_val, vmax_val, num=num_ticks)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f'{t:.2f}' for t in ticks])

    plt.show()
    
    
    
def display_predictions(model, test_loader, device):
    model.to(device)
    model.eval()

    class_indices = {i: [] for i in range(10)}
    
    for idx, (_, label) in enumerate(test_loader.dataset):
        class_indices[label].append(idx)
        
    random_indices = [random.choice(indices) for indices in class_indices.values()]
    
    sample_images = torch.stack([test_loader.dataset[i][0] for i in random_indices])
    sample_labels = [test_loader.dataset[i][1] for i in random_indices]

    with torch.no_grad():
        outputs = model(sample_images.to(device))
        _, predictions = torch.max(outputs, 1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('Model Predictions for a Sample of Each Class', fontsize=16)

    for i, ax in enumerate(axes.flat):
        image = sample_images[i].cpu().squeeze()
        true_label = sample_labels[i]
        predicted_label = predictions[i].item()

        ax.imshow(image, cmap='gray')
        
        title_color = 'green' if true_label == predicted_label else 'red'
        ax.set_title(f"True: {true_label}\nPred: {predicted_label}", color=title_color)
        
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.3)
    plt.show()
    
    
    
def plot_metrics(train_loss, test_acc):
    num_epochs = len(train_loss)
    epochs = range(1, num_epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(epochs, train_loss, marker='o', linestyle='-', color='royalblue')
    ax1.set_title('Training Loss Over Epochs', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2.plot(epochs, test_acc, marker='o', linestyle='-', color='red')
    ax2.set_title('Test Accuracy Over Epochs', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()