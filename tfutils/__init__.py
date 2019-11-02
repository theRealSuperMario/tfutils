import tensorflow as tf
import numpy as np


def slice_img_around_mu(img, mu, slice_size):
    """ extract rectangular slice around img with mean mu
    :param img:
    :param mu: in range [-1, 1]
    :param slice_size: tuple of ints
    :return: bn, n_part, slice_size[0] , slice_size[1], channel colour + n_part
    """

    h, w = slice_size
    bn, img_h, img_w, c = img.get_shape().as_list()  # bn this actually 2bn now
    bn_2, nk, _ = mu.get_shape().as_list()
    assert int(h / 2)
    assert int(w / 2)
    assert bn_2 == bn

    scal = tf.constant([img_h, img_w], dtype=tf.float32)
    mu = tf.stop_gradient(mu)
    mu_no_grad = tf.einsum("bkj,j->bkj", (mu + 1) / 2.0, scal)
    mu_no_grad = tf.cast(mu_no_grad, dtype=tf.int32)

    mu_no_grad = tf.reshape(mu_no_grad, shape=[bn, nk, 1, 1, 2])
    y = tf.tile(
        tf.reshape(tf.range(-h // 2, h // 2), [1, 1, h, 1, 1]), [bn, nk, 1, w, 1]
    )
    x = tf.tile(
        tf.reshape(tf.range(-w // 2, w // 2), [1, 1, 1, w, 1]), [bn, nk, h, 1, 1]
    )

    field = tf.concat([y, x], axis=-1) + mu_no_grad

    h1 = tf.tile(tf.reshape(tf.range(bn), [bn, 1, 1, 1, 1]), [1, nk, h, w, 1])

    idx = tf.concat([h1, field], axis=-1)

    image_slices = tf.gather_nd(img, idx)
    image_slices = tf.transpose(image_slices, perm=[0, 2, 3, 1, 4])
    return image_slices


def slice_img_with_mu_L_inv(img, mu, L_inv, scale=1.0, threshold=0.6):
    """slices images with ellipses centered around means (mu) and inverse L matrices.
    :param img: batch of images [N, H, W, C]
    :param mu:  batch of part means in range [-1, 1] : [N, P, 2]
    :param L_inv: L_inverses [N, P, 2, 2]
    :param scale: float scalar to scale L_inv with
    :param threshold: threshold to scale
    :return: slice [N, H, W, P, C]
    """
    bn, h, w, nc = img.get_shape().as_list()
    bn, nk, _ = mu.get_shape().as_list()

    mu_stop = tf.stop_gradient(mu)

    y_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, h), [h, 1]), [1, w])
    x_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, w), [1, w]), [h, 1])
    x_t_flat = tf.reshape(x_t, (1, 1, -1))
    y_t_flat = tf.reshape(y_t, (1, 1, -1))

    mesh = tf.concat([y_t_flat, x_t_flat], axis=-2)
    dist = mesh - tf.expand_dims(mu_stop, axis=-1)

    proj_precision = (
        tf.einsum("bnik,bnkf->bnif", scale * L_inv, dist) ** 2
    )  # tf.matmul(precision, dist)**2
    proj_precision = tf.reduce_sum(proj_precision, axis=-2)  # sum x and y axis

    heat = 1 / (1 + proj_precision)

    heat = tf.reshape(heat, shape=[bn, nk, h, w])  # bn width height number parts
    heat_scal = tf.clip_by_value(t=heat, clip_value_min=0.0, clip_value_max=1.0)
    mask = tf.where(
        heat_scal > threshold,
        tf.ones_like(heat_scal, dtype=img.dtype),
        tf.zeros_like(heat_scal, dtype=img.dtype),
    )  # [N, kp, H, W]
    mask = tf.transpose(mask, perm=[0, 2, 3, 1])
    slice_ = tf.expand_dims(img, axis=-2) * tf.expand_dims(mask, axis=-1)
    return slice_


def appearance_augmentation(
    t, contrast_var=0.3, brightness_var=0.3, saturation_var=0.3, hue_var=0.3
):
    """t in range [0, 1]"""
    t = tf.image.random_contrast(t, lower=1 - contrast_var, upper=1 + contrast_var)
    t = tf.image.random_brightness(t, brightness_var)
    t = tf.image.random_saturation(t, 1 - saturation_var, 1 + saturation_var)
    t = tf.image.random_hue(t, max_delta=hue_var)
    return t


def augment_img_at_mask(img, mask, augm_function=appearance_augmentation):
    """mask
    :param img: batch of images [N, H, W, C]
    :param mask: mask [N, H, W, P]
    :return: slice [N, H, W, C]
    # TODO: test
    """
    blending_map = mask
    augmented = tf.stack(
        [
            augm_function(img, contrast_var=0.3, saturation_var=0.3, hue_var=0.3)
            for i in range(mask.shape.as_list()[-1])
        ],
        axis=3,
    )
    augmented *= tf.expand_dims(blending_map, axis=4)
    return tf.reduce_sum(augmented, axis=3, keep_dims=False)
