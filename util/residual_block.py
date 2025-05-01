from tensorflow.keras import (
    layers,
    activations,
)

def ResidualBlock(width):
    """Create a residual block for a convolutional neural network.

    This function returns a callable that applies a residual block to an input tensor.
    The block consists of batch normalization, two convolutional layers, and a skip
    connection. If the input and output channel dimensions differ, a 1x1 convolution
    aligns the dimensions before addition.

    NOTE:
        - Uses the Swish activation function between convolutions.
        - Commonly used in U-Net and ResNet architectures for stable training.

    Args:
        width (int): Number of output channels for the block.

    Returns:
        Callable[[tf.Tensor], tf.Tensor]: Function that applies the residual block to an input tensor.
    """
    def apply(x):
        input_width = x.shape[3]

        # Check if the number of channels in the input matches the number of channels
        # that we would like the block to output. If not, include an extra `Conv2D` layer on
        # the skip connection to bring the number of channels in line with the rest of the
        # block.
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x) 

        # Apply a `BatchNormalization` layer.
        x = layers.BatchNormalization(center=False, scale=False)(x)

        # Apply a pair of Conv2D layers with a ReLU activation function in between.
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=activations.swish
        )(x)

        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)

        # Add the original block input to the output to provide the final output from the block.
        x = layers.Add()([x, residual])

        return x

    return apply


def DownBlock(width, block_depth):
    """Create a downsampling block for a U-Net architecture.

    This function returns a callable that applies a sequence of residual blocks followed
    by average pooling for downsampling. Intermediate outputs are saved for skip connections.

    NOTE:
        - Each residual block increases the number of channels.
        - Outputs for skip connections are appended to the `skips` list.

    Args:
        width (int): Number of output channels for the residual blocks.
        block_depth (int): Number of residual blocks to apply before downsampling.

    Returns:
        Callable[[Tuple[tf.Tensor, List[tf.Tensor]]], Tuple[tf.Tensor, List[tf.Tensor]]]:
            Function that applies the downsampling block.
    """
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            # The DownBlock increases the number of channels in the image using a Residual
            # Block of a given width…
            x = ResidualBlock(width)(x)
            # …each of which are saved to a list (skips) for use later by the UpBlocks.
            skips.append(x)
        # …and then applies a `AveragePooling2D` layer to reduce the spatial dimensions of the
        # image by a factor of 2.
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    """Create an upsampling block for a U-Net architecture.

    This function returns a callable that applies upsampling, concatenates skip connections,
    and applies a sequence of residual blocks to refine the feature maps.

    NOTE:
        - Upsamples input by a factor of 2 using bilinear interpolation.
        - Concatenates with corresponding skip connections from the encoder path.

    Args:
        width (int): Number of output channels for the residual blocks.
        block_depth (int): Number of residual blocks to apply after upsampling and concatenation.

    Returns:
        Callable[[Tuple[tf.Tensor, List[tf.Tensor]]], tf.Tensor]:
            Function that applies the upsampling block.
    """
    def apply(x):
        x, skips = x
        # The UpBlock begins with an UpSampling2D layer that doubles the size of the image.
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            # The output from a DownBlock layer is glued to the current output using a
            # `Concatenate` layer.
            x = layers.Concatenate()([x, skips.pop()])
            # A ResidualBlock is used to reduce the number of channels in the image as it
            # passes through the UpBlock.
            x = ResidualBlock(width)(x)
        return x

    return apply
