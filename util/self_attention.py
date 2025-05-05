import tensorflow as tf
from tensorflow.keras import layers

class SelfAttention(layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = layers.Conv2D(channels, 1)
        self.key = layers.Conv2D(channels, 1)
        self.value = layers.Conv2D(channels, 1)
        
    def call(self, x):
        # Get tensor shape using tf.shape
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        
        # Create query/key/value projections
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for attention computation
        q_reshaped = tf.reshape(q, [batch_size, height * width, self.channels])
        k_reshaped = tf.reshape(k, [batch_size, height * width, self.channels])
        v_reshaped = tf.reshape(v, [batch_size, height * width, self.channels])
        
        # Compute attention weights
        scale = tf.math.sqrt(tf.cast(self.channels, tf.float32))
        attention = tf.matmul(q_reshaped, k_reshaped, transpose_b=True) / scale
        attention = tf.nn.softmax(attention, axis=-1)
        
        # Apply attention to values
        output = tf.matmul(attention, v_reshaped)
        output = tf.reshape(output, [batch_size, height, width, self.channels])
        
        return x + output  # Residual connection
