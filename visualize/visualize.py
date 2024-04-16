import numpy as np
import matplotlib.pyplot as plt
import os

def create_save_directory(info):
    """
    Creates a directory for saving images if it does not already exist.
    :param info: A tuple containing (medical_type, model_type) to define the directory.
    :return: The filepath where the images will be saved.
    """
    medical_type, model_type, task = info
    directory = os.path.join("results","visualization", task, medical_type, model_type, "images")
    filename = "cxr_images.png"
    filepath = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)
    return filepath

def select_random_images(generator, num_images=2, mean_and_std = False):
    """
    Selects a specified number of random images from a data generator.
    :param generator: The data generator from which to draw images.
    :param num_images: The number of random images to select. Default is 2.
    :return: A list of randomly selected images.
    """
    images, labels = next(generator)
    selected_images = []
    selected_labels = []
    for _ in range(num_images):
        idx = np.random.randint(0, images.shape[0])
        image = images[idx]
        label = labels[idx]
        if image.shape[0] < image.shape[-1]:
            image = image.transpose(1, 2, 0)
            if mean_and_std:
                image = image * mean_and_std["std"] + mean_and_std["mean"]
                image = image.clip(0, 1)
        selected_images.append(image)
        selected_labels.append(label)
    generator.reset()
    return selected_images, selected_labels

def plot_images(set_images_labels, titles, filepath):
    """
    Plots a list of images and saves them to a file.
    :param images: A list of lists of images to plot.
    :param titles: The titles for each subplot.
    :param filepath: The filepath where the plot will be saved.
    :return: None.
    """
    plt.figure(figsize=(12, 8))
    for i, (image_batch,label_batch) in enumerate(set_images_labels):
        for j, (image,label) in enumerate(zip(image_batch,label_batch)):
            plt.subplot(3, len(image_batch), j + 1 + (i * len(image_batch)))
            plt.imshow(image)
            label = "Covid" if label == 1 else "Normal"
            plt.title(f"Patient has {label} lungs.", fontsize = 18)
            plt.axis('off')  
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Results saved to {filepath}")
    plt.close()

def save_random_images_from_generators(generators, info, num_images=2, mean_and_std = False):
    """
    Saves random images from data generators into a combined image file.
    :param generators: A list of data generators from which to draw images.
    :param info: A tuple containing (medical_type, model_type) to define the directory for saving images.
    :param num_images: The number of random images to save from each generator. Default is 2.
    :return: None.
    """
    filepath = create_save_directory(info)
    all_images = []
    for gen in generators:
        selected_images,selected_labels = select_random_images(gen, num_images, mean_and_std = mean_and_std)
        all_images.append((selected_images,selected_labels))
        gen.reset()
    plot_images(all_images, ["Train", "Validation", "Test"], filepath)