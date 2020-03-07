import inspect
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.ops import clip_ops


def _get_shape(x):
    if hasattr(x, 'dense_shape'):
        return x.dense_shape

    return K.shape(x)


def add_gradient_noise(BaseOptimizer):
    """
    Given a Keras-compatible optimizer class, returns a modified class that
    supports adding gradient noise as introduced in this paper:
    https://arxiv.org/abs/1511.06807
    The relevant parameters from equation 1 in the paper can be set via
    noise_eta and noise_gamma, set by default to 0.3 and 0.55 respectively.
    """
    if not (
        inspect.isclass(BaseOptimizer) and
        issubclass(BaseOptimizer, keras.optimizers.Optimizer)
    ):
        raise ValueError(
            'add_gradient_noise() expects a valid Keras optimizer'
        )

    class NoisyOptimizer(BaseOptimizer):
        def __init__(self, noise_eta=0.3, noise_gamma=0.55, **kwargs):
            super(NoisyOptimizer, self).__init__(**kwargs)
            with K.name_scope(self.__class__.__name__):
                self.noise_eta = K.variable(noise_eta, name='noise_eta')
                self.noise_gamma = K.variable(noise_gamma, name='noise_gamma')

            if "clipnorm" in kwargs:
                self.clipnorm = K.variable(kwargs.pop("clipnorm"), name="clipnorm")
            if "clipvalue" in kwargs:
                self.clipvalue = K.variable(kwargs.pop("clipvalue"), name="clipvalue")

        def get_gradients(self, loss, params):
            grads = super(NoisyOptimizer, self).get_gradients(loss, params)
            if hasattr(self, "clipnorm"):
                grads = [clip_ops.clip_by_norm(g, self.clipnorm) for g in grads]
            if hasattr(self, "clipvalue"):
                grads = [
                    clip_ops.clip_by_value(g, -self.clipvalue, self.clipvalue)
                    for g in grads
                ]
            # Add decayed gaussian noise
            t = K.cast(self.iterations, K.dtype(grads[0]))
            variance = self.noise_eta / ((1 + t) ** self.noise_gamma)

            grads = [
                grad + K.random_normal(
                    _get_shape(grad),
                    mean=0.0,
                    stddev=K.sqrt(variance),
                    dtype=K.dtype(grads[0])
                )
                for grad in grads
            ]

            return grads

        def get_config(self):
            config = {'noise_eta': float(K.get_value(self.noise_eta)),
                      'noise_gamma': float(K.get_value(self.noise_gamma))}
            base_config = super(NoisyOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    NoisyOptimizer.__name__ = 'Noisy{}'.format(BaseOptimizer.__name__)

    return NoisyOptimizer