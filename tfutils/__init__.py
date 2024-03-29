import tensorflow as tf
import numpy as np


def slice_img_around_mu(img, mu, slice_size):
    """extract rectangular slice around img center point `mu`
    
    Parameters
    ----------
    img : Tensor
        in range [-1, 1]. Shaped [N, H, W, C]
    mu : Tensor
        Batch of center points. Shaped [N, P, 2]
    slice_size : Tuple of ints
        height and width of slices to extract
    
    Returns
    -------
    image_slices
        Tensor of extracted image slices. Shaped [N, slice_size[0], slice_size[1], P, C]

    Examples
    --------
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

    References
    ----------
        Originally from https://github.com/CompVis/unsupervised-disentangling        
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


def mask_img_around_mu():
    """similar to @slice_img_around_mu, this function returns a masked version (same shape) of the input
    Parameters
    ----------
    img : Tensor
        in range [-1, 1]. Shaped [N, H, W, C]
    mu : Tensor
        Batch of center points. Shaped [N, P, 2]
    slice_size : Tuple of ints
        height and width of slices to extract
    
    Returns
    -------
    image_slices
        Tensor of extracted image slices. Shaped [N, slice_size[0], slice_size[1], P, C]

    Examples
    --------
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

    References
    ----------
        Originally from https://github.com/CompVis/unsupervised-disentangling
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
    
    Parameters
    ----------
    img : Tensor
        batch of images shaped [N, H, W, C]
    mu : Tensor
        batch of means in range [-1, 1]. shaped [N, P, 2]
    L_inv : Tensor
        batch of L_inverses [N, P, 2, 2]
    scale : float, optional
        scalar value to scale L_inv with, by default 1.0
    threshold : float, optional
        threshold of heatmap for cutoff, by default 0.6
    
    Returns
    -------
    sliced_image
        Stack of sliced images. Shaped [N, H, W, P, C]

    Examples
    --------
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

    References
    ----------
        Originally from https://github.com/CompVis/unsupervised-disentangling
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
    image, contrast_var=0.3, brightness_var=0.3, saturation_var=0.3, hue_var=0.3
):
    """Apply a range of appearance augmentations to an image tensor.
    
    Parameters
    ----------
    image : Tensor
        tensor of images. [N, H, W, C]
    contrast_var : float, optional
        maximum contrast variation, by default 0.3
    brightness_var : float, optional
        maximum brightness variation, by default 0.3
    saturation_var : float, optional
        maximum saturation variation, by default 0.3
    hue_var : float, optional
        maximum hue variation, by default 0.3
    
    Returns
    -------
    augmented:
        Augmented image tensor shaped [N, H, W, C]
        
    """
    augmented = tf.image.random_contrast(
        image, lower=1 - contrast_var, upper=1 + contrast_var
    )
    augmented = tf.image.random_brightness(augmented, brightness_var)
    augmented = tf.image.random_saturation(
        augmented, 1 - saturation_var, 1 + saturation_var
    )
    augmented = tf.image.random_hue(augmented, max_delta=hue_var)
    return augmented


def augment_img_at_mask(img, mask, augm_function=appearance_augmentation):
    """Augment image only as locations specified by mask tensor
    
    Parameters
    ----------
    img : Tensor
        image tensor shaped [N, H, W, C]
    mask : Tensor
        1 - hot mask tensor shaped [N, H, W, P]. Each slice along last axis is a boolean mask where to apply a certain augmentation
    augm_function : callable, optional
        function that yields augmented image tensors, by default appearance_augmentation
    
    Returns
    -------
    Tensor
        Tensor with augmented images shaped [N, H, W, C]

    Examples
    --------
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


from skimage import data


def astronaut(n_times):
    """return skimage.data.astronaut() n_times stacked in a batch of images
    
    Parameters
    ----------
    n_times : int
        how many times to stack along batch axis

    Examples
    --------
        image = astronaut(4)
        image.shape
        >>> [4, 512, 512, 3]
    """
    image = data.astronaut() / 255.0
    image = tf.to_float(image)
    return tf.stack([image] * n_times, axis=0)


def discriminator_patch(image, train=True):
    """ Discriminator on a patch of images with shape [49, 49]
    
    Parameters
    ----------
    image: Tensor
        batch of images shaped [N, 49, 49, C]
    train: bool, optional
        train flag for batch norm

    Returns
    -------
    probs, logits

    References
    ----------
        Originally from https://github.com/CompVis/unsupervised-disentangling
    """
    padding = "VALID"
    x0 = image
    x1 = tf.layers.conv2d(
        x0,
        32,
        4,
        strides=1,
        padding=padding,
        activation=tf.nn.leaky_relu,
        name="conv_0",
    )  # 46
    x1 = tf.layers.batch_normalization(x1, training=train, name="bn_0")
    x1 = tf.layers.conv2d(
        x1,
        64,
        4,
        strides=2,
        padding=padding,
        activation=tf.nn.leaky_relu,
        name="conv_1",
    )  # 44
    x1 = tf.layers.batch_normalization(x1, training=train, name="bn_1")
    x2 = tf.layers.conv2d(
        x1,
        128,
        4,
        strides=2,
        padding=padding,
        activation=tf.nn.leaky_relu,
        name="conv_2",
    )  # 10
    x2 = tf.layers.batch_normalization(x2, training=train, name="bn_2")
    x3 = tf.layers.conv2d(
        x2,
        256,
        4,
        strides=2,
        padding=padding,
        activation=tf.nn.leaky_relu,
        name="conv_3",
    )  # 4
    x3 = tf.layers.batch_normalization(x3, training=train, name="bn_3")
    x4 = tf.reshape(x3, shape=[-1, 4 * 4 * 256])
    x4 = tf.layers.dense(x4, 1, name="last_fc")
    return tf.nn.sigmoid(x4), x4


def _draw_rect(center, h, w, imsize):
    """Draw rectangle around center coord. Use @draw_rect for a batched version
    
    Parameters
    ----------
    center : tensor
        center coordinate. Shaped [2,]. Locations in pixels (integers).
    h : int
        Height of rectangles to draw
    w : int
        Width of rectangles to draw
    imsize : tuple, optional
        final output image size, by default (10, 10, 1)

    Examples
    --------
        center = tf.constant([5, 7])
        h = 3
        w = 3
        imsize = (10, 10, 3)
        m = _draw_rect(center, h, w, imsize)
        plt.imshow(np.squeeze(m))
    """
    cx, cy = tf.split(center, 2, axis=0)
    cx = tf.squeeze(cx)
    cy = tf.squeeze(cy)

    pad_sizeh = h
    pad_sizew = w
    cx += pad_sizeh
    cy += pad_sizew
    imsize_pad = (imsize[0] + 2 * pad_sizeh, imsize[1] + 2 * pad_sizew, imsize[2])

    indices = tf.meshgrid(
        tf.range(cx - h // 2, cx + h // 2), tf.range(cy - w // 2, cy + w // 2)
    )
    indices = tf.stack(indices, 0)
    indices = tf.reshape(indices, (2, tf.reduce_prod(tf.shape(indices)[1:])))

    indices_flat = tf_ravel_multi_index(indices, imsize_pad[:2])

    indices_flat = tf.reshape(indices_flat, (tf.shape(indices_flat)[0], 1))

    updates = tf.ones((tf.shape(indices_flat)[0], imsize_pad[-1]))
    shape = tf.constant([np.prod(imsize_pad[:2]), imsize_pad[-1]])
    rect = tf.scatter_nd(indices_flat, updates, shape)

    rect = tf.reshape(rect, imsize_pad)
    rect = tf.image.crop_to_bounding_box(
        rect, pad_sizeh, pad_sizew, imsize[0], imsize[1]
    )
    return rect


def draw_rect(center, h: int, w: int, imsize=(10, 10, 1)):
    """Draw rectangle filled with ones
    
    Parameters
    ----------
    center : tensor
        A batch of center coordinates for the rectangles. Shaped [N, 2]. Locations in pixels (integers).
    h : int
        Height of rectangles to draw
    w : int
        Width of rectangles to draw
    imsize : tuple, optional
        final output image size, by default (10, 10, 1)
    
    Returns
    -------
    rect
        batch of rectangles shaped [N,] + imsize

    Examples
    --------
        center = tf.constant([5, 7, 6, 6])
        center = tf.reshape(center, (2, 2))
        h = 3
        w = 3
        imsize = (10, 10, 1)
        m = draw_rect(center, h, 2, imsize)

        plt.imshow(np.squeeze(m[1, ...]))
    """
    rects = tf.map_fn(lambda x: _draw_rect(x, h, w, imsize), center, dtype=tf.float32)
    return rects


def tf_ravel_multi_index(multi_index, dims):
    # TODO: add link to stackoverflow
    """[summary]
    
    Returns
    -------
    [type]
        [description]

    Examples
    --------
        idx = tf_ravel_multi_index(np.array([[5, 0], [1, 1]]), (10, 10))
        i = np.zeros((100, ))
        i[np.array(idx)] = 1
        i = i.reshape((10, 10))

        plt.imshow(i)
    """
    strides = tf.cumprod(dims, exclusive=True, reverse=True)
    return tf.reduce_sum(multi_index * tf.expand_dims(strides, 1), axis=0)
