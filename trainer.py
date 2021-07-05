import tensorflow as tf
import glob
import imageio
import PIL
import pathlib
import matplotlib.pyplot as plt
import sys

import models


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python trainer.py epochs")

    EPOCHS = int(sys.argv[1])

    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data(
        path="mnist.npz"
    )
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(
        "float32"
    )
    # normalise to [-1, 1]
    train_images = (train_images - 127.5) / 127.5

    # Batch and shuffle the data
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(models.BUFFER_SIZE)
        .batch(models.BATCH_SIZE)
    )

    # make images dir if it doesnt exist
    pathlib.Path("./images").mkdir(exist_ok=True)

    # interactive plots
    plt.ion()
    plt.show()

    models.train(train_dataset, EPOCHS)

    anim_file = "mnistGan.gif"
    with imageio.get_writer(anim_file, mode="I") as writer:
        filenames = glob.glob("images/image*.png")
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open("image_at_epoch_{:04d}.png".format(epoch_no))


if __name__ == "__main__":
    main()
