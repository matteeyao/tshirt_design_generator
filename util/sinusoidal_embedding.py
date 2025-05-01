import tensorflow as tf
import math

NOISE_EMBEDDING_SIZE = 32

def sinusoidal_embedding(x):
    """Convert a scalar noise variance into a sinusoidal embedding vector.

    This function maps a single noise variance scalar (or a batch of scalars) into a
    high-dimensional embedding vector using sinusoidal functions. This is commonly used
    in diffusion models to encode the diffusion step or noise level as a vector of length 32.

    NOTE:
        - The embedding uses 16 different frequencies, producing a vector of length 32
          by concatenating sine and cosine components.
        - This approach is inspired by positional encodings in transformer models.

    Args:
        x (tf.Tensor): Input tensor of shape (..., 1, 1, 1), representing the noise variance(s).

    Returns:
        tf.Tensor: Sinusoidal embedding tensor of shape (..., 1, 1, 32).
    """

    # 1. Compute a range of frequencies spaced logarithmically between 1 and 1000 (16 frequencies).
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(1.0),
            tf.math.log(1000.0),
            NOISE_EMBEDDING_SIZE // 2,
        )
    )

    # 2. Convert frequencies to angular speeds for the sinusoidal functions.
    angular_speeds = 2.0 * math.pi * frequencies

    # 3. Compute the sinusoidal embeddings by concatenating sine and cosine of the scaled input.
    #    This results in a vector of length 32 for each scalar input.
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )

    # 4. Return the embedding tensor.
    return embeddings

