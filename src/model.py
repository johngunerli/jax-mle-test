from flax import nnx
import jax


class SimpleModel(nnx.Module):
    def __init__(self, num_classes: int, *, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(28 * 28, 512, rngs=rngs)
        self.dense2 = nnx.Linear(512, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs):  # <-- add rngs here
        x = nnx.relu(self.dense1(x))
        return self.dense2(x)
