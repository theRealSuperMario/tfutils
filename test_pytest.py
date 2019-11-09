import pytest
import numpy as np
import tensorflow as tf

from skimage import data
from matplotlib import pyplot as plt


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
    tf.enable_eager_execution()
    from tfutils import slice_img_around_mu

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
    tf.enable_eager_execution()
    from tfutils import appearance_augmentation

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
    tf.enable_eager_execution()
    from tfutils import slice_img_with_mu_L_inv

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
    tf.enable_eager_execution()
    from tfutils import augment_img_at_mask

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


@pytest.mark.mpl_image_compare
def test__draw_rect():
    tf.enable_eager_execution()
    from tfutils import _draw_rect

    fig, ax = plt.subplots(1, 1)
    center = tf.constant([5, 7])
    h = 3
    w = 3
    imsize = (10, 10, 3)
    m = _draw_rect(center, h, w, imsize)
    ax.imshow(np.squeeze(m), interpolation="nearest")
    return fig


@pytest.mark.mpl_image_compare
def test_draw_rect_noneager():
    from tensorflow.python.framework.ops import disable_eager_execution

    disable_eager_execution()
    from tfutils import draw_rect

    fig, ax = plt.subplots(1, 2)

    h = 3
    w = 3
    imsize = (10, 10, 1)
    center = tf.placeholder(tf.int32, (None, 2), name="center")
    mm = draw_rect(center, h, w, imsize)
    with tf.Session() as sess:
        m = sess.run(mm, {center: np.reshape(np.array([5, 7, 6, 6]), (2, 2))})
    ax[0].imshow(np.squeeze(m[0, ...]), interpolation="nearest")
    ax[0].set_title("center : {}".format(np.array(center[0, :])))
    ax[1].imshow(np.squeeze(m[1, ...]), interpolation="nearest")
    ax[1].set_title("center : {}".format(np.array(center[1, :])))
    return fig


@pytest.mark.mpl_image_compare
def test_draw_rect():
    from tfutils import draw_rect

    fig, ax = plt.subplots(1, 2)
    center = tf.constant([5, 7, 6, 6])
    center = tf.reshape(center, (2, 2))
    h = 3
    w = 3
    imsize = (10, 10, 1)
    m = draw_rect(center, h, w, imsize)
    ax[0].imshow(np.squeeze(m[0, ...]), interpolation="nearest")
    ax[0].set_title("center : {}".format(center[0, :]))
    ax[1].imshow(np.squeeze(m[1, ...]), interpolation="nearest")
    ax[1].set_title("center : {}".format(center[1, :]))
    return fig


@pytest.mark.mpl_image_compare
def test_draw_rect_noneager():
    from tensorflow.python.framework.ops import disable_eager_execution

    disable_eager_execution()
    from tfutils import draw_rect

    fig, ax = plt.subplots(1, 2)

    h = 3
    w = 3
    imsize = (10, 10, 1)
    center = tf.placeholder(tf.int32, (None, 2), name="center")
    mm = draw_rect(center, h, w, imsize)
    with tf.Session() as sess:
        m = sess.run(mm, {center: np.reshape(np.array([5, 7, 6, 6]), (2, 2))})
    ax[0].imshow(np.squeeze(m[0, ...]), interpolation="nearest")
    ax[0].set_title("center : {}".format(np.array(center[0, :])))
    ax[1].imshow(np.squeeze(m[1, ...]), interpolation="nearest")
    ax[1].set_title("center : {}".format(np.array(center[1, :])))
    return fig
