import pytest
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from skimage import data


def astronaut(n_times):
    """return skimage.data.astronaut() n_times stacked in a batch of images
    
    Parameters
    ----------
    n_times : int
        how many times to stack along batch axis
    """
    image = data.astronaut() / 255.0
    image = tf.to_float(image)
    return tf.stack([image] * n_times, axis=0)


@pytest.mark.mpl_image_compare
def test_slice_img_around_mu():
    from tfutils import slice_img_around_mu
    from matplotlib import pyplot as plt

    mu = np.zeros((1, 1, 2), dtype=np.float32)
    mu = tf.concat([mu, mu + 0.25], axis=1)
    mut = tf.convert_to_tensor(mu)
    imgt = astronaut(1)

    slice_ = slice_img_around_mu(imgt, mut, (200, 200))

    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    ax[0].imshow(np.squeeze(imgt)[...])
    ax[0].set_title("Image")
    ax[1].imshow(np.squeeze(slice_)[:, :, 0, :])
    ax[1].set_title("slice")
    ax[2].imshow(np.squeeze(slice_)[:, :, 1, :])
    ax[2].set_title("slice")
    return fig


@pytest.mark.mpl_image_compare
def test_appearance_augmentation():
    from tfutils import appearance_augmentation
    from matplotlib import pyplot as plt

    tf.set_random_seed(40)
    imgt = astronaut(2)
    augmented = appearance_augmentation(imgt)

    # fig, ax = plt.subplots(2, 2)
    # ax = ax.ravel()
    # ax[0].imshow(np.squeeze(imgt)[...])
    # ax[0].set_title("Image")
    # ax[1].imshow(np.squeeze(slice_)[:, :, 0, :])
    # ax[1].set_title("slice")
    # ax[2].imshow(np.squeeze(slice_)[:, :, 1, :])
    # ax[2].set_title("slice")

    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    ax[0].imshow(np.squeeze(imgt)[0, ...])
    ax[0].set_title("Image 1")
    ax[1].imshow(np.squeeze(augmented)[0, ...])
    ax[1].set_title("Augmented image 1")
    ax[2].imshow(np.squeeze(imgt)[1, ...])
    ax[2].set_title("Image 2")
    ax[3].imshow(np.squeeze(augmented)[1, ...])
    ax[3].set_title("Augmented image 2")
    return fig


@pytest.mark.mpl_image_compare
def test_slice_img_with_mu_L_inv():
    from tfutils import slice_img_with_mu_L_inv
    from matplotlib import pyplot as plt

    mu = np.zeros((1, 1, 2), dtype=np.float32)
    mu = tf.concat([mu, mu + 0.25], axis=1)
    mut = tf.convert_to_tensor(mu)

    L_invt = tf.convert_to_tensor(np.array([[2, 0], [0, 1]], dtype=np.float32))
    L_invt = tf.expand_dims(L_invt, 0)
    L_invt = tf.expand_dims(L_invt, 0)
    L_invt = tf.concat([L_invt, 0.3 * L_invt], axis=1)

    imgt = astronaut(1)
    slice_ = slice_img_with_mu_L_inv(imgt, mut, L_invt, 3, 0.6)

    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    ax[0].imshow(np.squeeze(imgt)[...])
    ax[0].set_title("Image")
    ax[1].imshow(np.squeeze(slice_)[:, :, 0, :])
    ax[1].set_title("slice")
    ax[2].imshow(np.squeeze(slice_)[:, :, 1, :])
    ax[2].set_title("slice")
    return fig


@pytest.mark.mpl_image_compare
def test_augment_img_at_mask():
    from tfutils import augment_img_at_mask
    from matplotlib import pyplot as plt

    tf.set_random_seed(42)

    imgt = astronaut(1)
    mask = np.zeros((1, 512, 512, 2), dtype=np.float32)
    mask[:, :250, :, 0] = 1.0
    mask[:, 250:, :, 1] = 1.0

    a = augment_img_at_mask(imgt, tf.convert_to_tensor(mask))
    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    ax[0].imshow(np.squeeze(imgt))
    ax[0].set_title("image")
    ax[1].imshow(np.squeeze(a[0, ...]))
    ax[1].set_title("augmented")
    ax[2].imshow(np.squeeze(mask[..., 0]), cmap=plt.cm.gray)
    ax[2].set_title("mask 0")
    ax[3].imshow(np.squeeze(mask[..., 1]), cmap=plt.cm.gray)
    ax[3].set_title("mask 1")
    return fig
