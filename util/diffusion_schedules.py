import math
import tensorflow as tf

def linear_diffusion_schedule(diffusion_times):
    """Compute noise and signal rates using a linear diffusion schedule.

    This function implements a linear schedule for the diffusion process, where the noise
    rate (beta) increases linearly from `min_rate` to `max_rate` over the diffusion times.
    The function returns the corresponding noise and signal rates for each diffusion time.

    NOTE:
        - The schedule is commonly used in denoising diffusion models.
        - The returned rates are suitable for use in forward and reverse diffusion steps.

    Args:
        diffusion_times (tf.Tensor): Tensor of diffusion times, typically in [0, 1].

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple (noise_rates, signal_rates), both tensors of the same shape as diffusion_times.
    """
    min_rate = 0.0001
    max_rate = 0.02
    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1 - betas
    alpha_bars = tf.math.cumprod(alphas)
    signal_rates = tf.sqrt(alpha_bars)
    noise_rates = tf.sqrt(1 - alpha_bars)
    return noise_rates, signal_rates


def cosine_diffusion_schedule(diffusion_times):
    """Compute noise and signal rates using a cosine diffusion schedule.

    This function implements a cosine-based schedule for the diffusion process,
    where the signal and noise rates are computed as the cosine and sine of
    the scaled diffusion times, respectively.

    NOTE:
        - This schedule produces a smoother transition of noise and signal rates.
        - The returned rates can be used for both forward and reverse diffusion.

    Args:
        diffusion_times (tf.Tensor): Tensor of diffusion times, typically in [0, 1].

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple (noise_rates, signal_rates), both tensors of the same shape as diffusion_times.
    """
    signal_rates = tf.cos(diffusion_times * math.pi / 2)
    noise_rates = tf.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(diffusion_times):
    """Compute noise and signal rates using an offset cosine diffusion schedule.

    This function implements a modified cosine schedule for the diffusion process,
    where the signal rate is constrained between `min_signal_rate` and `max_signal_rate`.
    The diffusion angle is interpolated between the arccos of these rates.

    NOTE:
        - This schedule allows for more flexible control over the start and end signal rates.
        - The returned rates are used in advanced diffusion models for improved training dynamics.

    Args:
        diffusion_times (tf.Tensor): Tensor of diffusion times, typically in [0, 1].

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple (noise_rates, signal_rates), both tensors of the same shape as diffusion_times.
    """
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)

    return noise_rates, signal_rates

