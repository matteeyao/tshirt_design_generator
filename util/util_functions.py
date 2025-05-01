import matplotlib.pyplot as plt

def sample_batch(dataset):
    """Sample a single batch of images from a TensorFlow dataset.

    This function takes a TensorFlow dataset (such as one created by `tf.data.Dataset`)
    and returns a single batch of images as a NumPy array. If the dataset yields tuples
    (e.g., (images, labels)), only the images are returned.

    NOTE: The function assumes that the dataset yields either images or (images, labels)
    and that the batch size is set when batching the dataset.

    Args:
        dataset (tf.data.Dataset): A TensorFlow dataset yielding batched image data.

    Returns:
        numpy.ndarray: A batch of images as a NumPy array.
    """

    # 1. Take one batch from the dataset
    batch = dataset.take(1).get_single_element()

    # 2. If the batch is a tuple (e.g., (images, labels)), select only the images
    if isinstance(batch, tuple):
        batch = batch[0]

    # 3. Convert the batch to a NumPy array and return
    return batch.numpy()


def display(
    images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None
):
    """
    Displays n random images from each one of the supplied arrays.
    """
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    plt.figure(figsize=size)
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].astype(as_type), cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()

def display(
    images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None
):
    """Displays n random images from each one of the supplied arrays using matplotlib.

    This function visualizes `n` images from the provided array, normalizing the pixel values
    if necessary, and arranges them in a single row. Optionally, the figure can be saved
    to disk.

    NOTE:
        - If image values are greater than 1.0, they are assumed to be in [0, 255] and are scaled to [0, 1].
        - If image values are less than 0.0, they are assumed to be in [-1, 1] and are shifted to [0, 1].
        - The function displays the images using the specified colormap and data type.

    Args:
        images (numpy.ndarray): Array of images to display. Should be at least `n` images.
        n (int, optional): Number of images to display. Defaults to 10.
        size (tuple, optional): Figure size for matplotlib. Defaults to (20, 3).
        cmap (str, optional): Colormap to use for displaying images. Defaults to "gray_r".
        as_type (str, optional): Data type to cast images before displaying. Defaults to "float32".
        save_to (str, optional): If specified, saves the figure to this file path. Defaults to None.

    Returns:
        None
    """

    # 1. Normalize images if necessary
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    # 2. Create the figure and plot each image
    plt.figure(figsize=size)
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].astype(as_type), cmap=cmap)
        plt.axis("off")

    # 3. Save the figure if a path is provided
    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    # 4. Show the figure
    plt.show()
