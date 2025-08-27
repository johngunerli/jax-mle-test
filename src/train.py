import jax
import jax.numpy as jnp
import optax
from flax import nnx
from .model import SimpleModel
from .utils import get_dataloader
import onnx
from jax2onnx import to_onnx
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

os.environ["JAX_PLATFORM_NAME"] = "cpu"


def train():
    # Hyperparameters
    learning_rate = 0.01
    num_epochs = 1
    batch_size = 64
    num_classes = 10

    # Tensorboard writer
    writer = SummaryWriter("runs/mnist_experiment")

    # Create a dummy dataloader
    train_loader = get_dataloader(batch_size)

    # Initialize model and optimizer
    rngs = nnx.Rngs(0)
    model = SimpleModel(num_classes=num_classes, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = jnp.array(data.numpy().reshape(data.shape[0], -1))
            target_labels = jnp.array(target.numpy())
            target = jax.nn.one_hot(target_labels, num_classes)

            def loss_fn(model: SimpleModel):
                logits = model(data, rngs=rngs)
                return optax.softmax_cross_entropy(logits, target).mean()

            grad_fn = nnx.value_and_grad(loss_fn)
            loss, grads = grad_fn(model)
            optimizer.update(model, grads)

            if batch_idx % 100 == 0:
                writer.add_scalar(
                    "training loss",
                    np.asarray(loss),
                    epoch * len(train_loader) + batch_idx,
                )

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}")

    # Save the model using jax2onnx with a dynamic batch dimension, binding rngs
    rngs_export = nnx.Rngs(0)

    def export_fn(x):
        return model(x, rngs=rngs_export)

    onnx_model = to_onnx(export_fn, [("B", 28 * 28)])
    onnx.save_model(onnx_model, "model.onnx")

    print("Model saved to model.onnx")
    writer.close()


if __name__ == "__main__":
    train()
