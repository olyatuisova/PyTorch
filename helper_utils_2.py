import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torchvision.transforms import functional as F


letter_ref = [
    "Dear Laurence",
    "Hope the PyTorch course is going well",
    "Do notforget to keep the labs interesting and engaging",
    "Maybe the students could decode my messy handwriting",
    "That might be a bit too challenging though",
    "I am impressed you are able to read this",
]


path_data = "./EMNIST_data"


def load_hidden_message_images(file_name="hidden_message_images.pkl"):
    """
    Loads hidden message images from a pickle file.

    Args:
        file_name (str): The name of the file to load the images from.

    Returns:
        message_imgs (list): A list containing the loaded message images.
    """
    with open(file_name, "rb") as f:
        import pickle
        message_imgs = pickle.load(f)
    return message_imgs


def decode_word_imgs(word_imgs, model, device):
    """
    Decodes a sequence of character images into a single word string using a 
    provided classification model.

    Args:
        word_imgs (list): A collection of image tensors representing 
            individual characters in a word.
        model (torch.nn.Module): The trained neural network model used to 
            predict the character from the image.
        device (torch.device): The computation device to which the tensors 
            should be moved before inference.

    Returns:
        decoded_word (str): The concatenated string of predicted characters 
            forming the complete word.
    """
    model.eval()
    decoded_chars = []
    with torch.no_grad():
        for char_img in word_imgs:
            char_img = char_img.unsqueeze(0).to(device)
            output = model(char_img)
            _, predicted = output.max(1)
            predicted_label = predicted.item()
            lowercase_char = chr(ord("a") + predicted_label)
            decoded_chars.append(f"{lowercase_char}")
    decoded_word = "".join(decoded_chars)
    return decoded_word


def visualize_image(img, label=None, ax=None):
    """
    Visualizes an EMNIST image with its label. If an axis is provided, 
    plots on that axis; otherwise, creates a new figure.

    Args:
        img (np.ndarray): The image array to display.
        label (int): The numeric EMNIST label. If None, no title is shown.
        ax (matplotlib.axes.Axes): Axis to plot on. If None, creates a 
            new figure.
    """
    if isinstance(img, torch.Tensor):
        img = img.numpy().squeeze()
    elif isinstance(img, np.ndarray):
        if img.ndim == 3:
            img = img[:, :, 0]

    if label is not None:
        uppercase_char, lowercase_char = convert_emnist_label_to_char(label)
        title = f"EMNIST Letter: {uppercase_char}/{lowercase_char}"
    else:
        title = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        show_colorbar = True
    else:
        show_colorbar = False

    im = ax.imshow(img, cmap="gray")
    ax.set_xticks(np.arange(0, 28, 1))
    ax.set_yticks(np.arange(0, 28, 1))
    ax.tick_params(labelsize=6)
    ax.grid(True, color="red", alpha=0.3)

    if title:
        ax.set_title(title)

    if show_colorbar:
        plt.colorbar(im, ax=ax)
        plt.show()


def display_data_loader_contents(data_loader):
    """
    Displays the contents of the data loader including sizes and shapes.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader to 
            display.
    """
    try:
        print("Total number of images in dataset:", len(data_loader.dataset))
        print("Total number of batches:", len(data_loader))
        for batch_idx, (data, labels) in enumerate(data_loader):
            print(f"--- Batch {batch_idx + 1} ---")
            print(f"Data shape: {data.shape}")
            print(f"Labels shape: {labels.shape}")
            break
    except StopIteration:
        print("data loader is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")


def evaluate_per_class(model, test_loader, device):
    """
    Evaluates the model's accuracy for each individual class.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the 
            test dataset.
        device (torch.device): Device to run the model on.

    Returns:
        class_accuracies (dict): A dictionary containing accuracy for each 
            class letter.
    """
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets - 1
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    class_accuracies = {}

    for class_idx in range(26):
        class_targets = [
            t for t, p in zip(all_targets, all_predictions) if t == class_idx
        ]
        class_predictions = [
            p for t, p in zip(all_targets, all_predictions) if t == class_idx
        ]

        if len(class_targets) > 0:
            class_accuracies[chr(65 + class_idx)] = accuracy_score(
                class_targets, class_predictions
            )
        else:
            class_accuracies[chr(65 + class_idx)] = 0.0

    return class_accuracies


def save_student_model(model, filename="trained_student_model.pth"):
    """
    Saves the student's trained model and metadata to a file.

    Args:
        model (torch.nn.Module): The student's trained model.
        filename (str): The filename to save to.
    """
    save_dict = {"model": model}
    torch.save(save_dict, filename)
    print(f"Model saved to {filename}")


def convert_emnist_label_to_char(label):
    """
    Converts an EMNIST label to corresponding uppercase and lowercase letters.

    Args:
        label (int): The numeric EMNIST label.

    Returns:
        char_tuple (tuple): A tuple containing the uppercase and lowercase 
            characters.
    """
    if not (1 <= label <= 26):
        raise ValueError("Label must be between 1 and 26 inclusive.")

    uppercase_char = chr(64 + label)
    lowercase_char = chr(96 + label)
    char_tuple = (uppercase_char, lowercase_char)
    return char_tuple