import tensorflow as tf
from tensorflow.keras import layers

class SelfAttention(layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = layers.BatchNormalization()
        self.query = layers.Conv2D(channels, 1)
        self.key = layers.Conv2D(channels, 1)
        self.value = layers.Conv2D(channels, 1)
        
    def call(self, x):
        batch_size, height, width, _ = tf.shape(x)
        
        # Normalize inputs
        x_norm = self.norm(x)
        
        # Create query/key/value projections
        q = self.query(x_norm)
        k = self.key(x_norm)
        v = self.value(x_norm)
        
        # Reshape for attention computation
        q = tf.reshape(q, [-1, height * width, self.channels])
        k = tf.reshape(k, [-1, height * width, self.channels])
        v = tf.reshape(v, [-1, height * width, self.channels])
        
        # Compute attention weights
        attention = tf.matmul(q, k, transpose_b=True)
        attention = tf.nn.softmax(attention / tf.math.sqrt(tf.cast(self.channels, tf.float32)), axis=-1)
        
        # Apply attention to values
        out = tf.matmul(attention, v)
        out = tf.reshape(out, [-1, height, width, self.channels])
        
        return x + out  # Residual connection
