import tensorflow as tf
from tensorflow.keras import (
    layers,
    models,
    metrics,
)
from util.diffusion_schedules import offset_cosine_diffusion_schedule

class DiffusionModel(models.Model):
    """A Keras Model implementing a denoising diffusion probabilistic model (DDPM/UNet).

    This class encapsulates the logic for training and inference of a diffusion model,
    including normalization, noise scheduling, EMA (exponential moving average) of weights,
    and forward/reverse diffusion processes.

    Args:
        unet (tf.keras.Model): The U-Net model used for noise prediction during denoising.

    Attributes:
        normalizer (tf.keras.layers.Normalization): Layer for input normalization.
        network (tf.keras.Model): The primary U-Net network.
        ema_network (tf.keras.Model): Exponential Moving Average copy of the U-Net.
        diffusion_schedule (callable): Function to compute noise/signal rates per timestep.

    """
    def __init__(self, image_size, batch_size, unet, ema_rate):
        """Initialize the DiffusionModel.

        Args:
            unet (tf.keras.Model): The U-Net architecture for noise prediction.
            image_size (int): The size of the images to be generated.
            batch_size (int): The batch size for training.
            ema_rate (float): The rate for the exponential moving average.
        """
        super().__init__()

        self.normalizer = layers.Normalization()
        self.image_size = image_size
        self.batch_size = batch_size
        self.network = unet
        self.ema_rate = ema_rate
        self.ema_network = models.clone_model(self.network)
        self.diffusion_schedule = offset_cosine_diffusion_schedule

    def compile(self, **kwargs):
        """Configure the model for training.

        Initializes the noise loss tracker for monitoring training loss.

        Args:
            **kwargs: Additional keyword arguments for the parent compile method.
        """
        super().compile(**kwargs)
        self.noise_loss_tracker = metrics.Mean(name="n_loss")

    @property
    def metrics(self):
        """List of model metrics.

        Returns:
            list: List containing the noise loss tracker metric.
        """
        return [self.noise_loss_tracker]

    def denormalize(self, images):
        """Reverse the normalization applied to images.

        Converts images from normalized space (zero mean, unit variance)
        back to the original [0, 1] range.

        Args:
            images (tf.Tensor): Normalized images.

        Returns:
            tf.Tensor: Denormalized images, clipped to [0, 1].
        """
        # Generate some initial noise maps.
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        """Predict noise and reconstruct images from noisy inputs.

        Uses the network (or EMA network during inference) to predict the noise
        component in the input, and reconstructs the denoised images.

        Args:
            noisy_images (tf.Tensor): Batch of noisy images.
            noise_rates (tf.Tensor): Noise rates for each image.
            signal_rates (tf.Tensor): Signal rates for each image.
            training (bool): Whether to use the training or EMA network.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Predicted noise and reconstructed images.
        """
        if training:
            network = self.network
        else:
            network = self.ema_network
        pred_noises = network(
            [noisy_images, noise_rates**2], training=training
        )
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        """Iteratively denoise random noise to generate images.

        Runs the reverse diffusion process, starting from pure noise, and
        applies the denoising network at each step to reconstruct images.

        Args:
            initial_noise (tf.Tensor): Batch of initial noise images.
            diffusion_steps (int): Number of reverse diffusion steps.

        Returns:
            tf.Tensor: Batch of generated images.
        """
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise

        # Look over a fixed number of steps (e.g., 20).
        for step in range(diffusion_steps):
            # The diffusion times are all set to 1 (i.e., at the start of the reverse diffusion
            # process).
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            # The noise and signal rates are calculated according to the diffusion schedule.
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            # The U-Net is used to predict the noise, allowing us to calculate the denoised
            # image estimate.
            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates, training=False
            )
            # The next diffusion time is calculated by subtracting the step size from the
            # current diffusion time.
            # The next noise and signal rates are calculated according to the diffusion schedule.
            next_diffusion_times = diffusion_times - step_size
            #The new noise and signal rates are calculated.
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            # The t-1 images are calculated by reapplying the predicted noise to the predicted
            # image, according to the t-1 diffusion schedule rates.
            current_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )

        # After 20 steps, the final x_{0} predicted images are returned.
        return pred_images

    def generate(self, num_images, diffusion_steps, initial_noise=None):
        """Generate new images by running the reverse diffusion process.

        Starts from random Gaussian noise (or provided initial noise) and
        iteratively denoises to produce new synthetic images.

        Args:
            num_images (int): Number of images to generate.
            diffusion_steps (int): Number of reverse diffusion steps.
            initial_noise (tf.Tensor, optional): Custom initial noise tensor.
                If None, random noise is used.

        Returns:
            tf.Tensor: Batch of generated images.
        """
        if initial_noise is None:
            initial_noise = tf.random.normal(
                shape=(num_images, self.image_size, self.image_size, 3)
            )
        
        # Apply the reverse diffusion process.
        generated_images = self.reverse_diffusion(
            initial_noise, diffusion_steps
        )

        # The images output by the network will have mean zero and unit variance, so we
        # need to denormalize by reapplying the mean and variance calculated from the
        # training data.
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        """Perform a single training step.

        Normalizes the input images, adds noise, computes the loss, applies gradients,
        and updates the exponential moving average (EMA) network.

        Args:
            images (tf.Tensor): Batch of input images.

        Returns:
            dict: Dictionary of metric names and values.
        """
        # We first normalize the batch of images to have zero mean and unit variance.
        images = self.normalizer(images, training=True)
        # Next, we sample noise to match the shape of the input images.
        noises = tf.random.normal(shape=(self.batch_size, self.image_size, self.image_size, 3))

        # We also sample random diffusion times…
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )

        # …and use these to generate the noise and signal rates according to the cosine dif‐
        # fusion schedule.
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        # Then we apply the signal and noise weightings to the input images to generate
        # the noisy images.
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # Next, we denoise the noisy images by asking the network to predict the noise and
            # then undoing the noising operation, using the provided noise_rates and
            # signal_rates.
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            # We can then calculate the loss (mean absolute error) between the predicted noise
            # and the true noise…
            noise_loss = self.loss(noises, pred_noises)  # used for training

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights)
        )

        # …and take a gradient step against this loss function.
        self.noise_loss_tracker.update_state(noise_loss)

        # …and update the EMA weights.
        for weight, ema_weight in zip(
            self.network.weights, self.ema_network.weights
        ):
            # The EMA network weights are updated to a weighted average of the existing
            # EMA weights and the trained network weights after the gradient step.
            ema_weight.assign(self.ema_rate * ema_weight + (1 - self.ema_rate) * weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        """Perform a single validation/test step.

        Normalizes the input images, adds noise, predicts and computes the loss,
        and updates the noise loss tracker.

        Args:
            images (tf.Tensor): Batch of input images.

        Returns:
            dict: Dictionary of metric names and values.
        """
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(self.batch_size, self.image_size, self.image_size, 3))
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )
        noise_loss = self.loss(noises, pred_noises)
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}
