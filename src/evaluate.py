import onnxruntime as ort
from .utils import get_dataloader
import numpy as np


def evaluate():
    # Load the ONNX model
    ort_session = ort.InferenceSession("model.onnx")

    # Create a dummy dataloader
    test_loader = get_dataloader(batch_size=64, train=False)

    correct = 0
    total = 0
    for data, target in test_loader:
        data_np = data.numpy().reshape(data.shape[0], -1)
        target_np = target.numpy()

        ort_inputs = {ort_session.get_inputs()[0].name: data_np}
        ort_outs = ort_session.run(None, ort_inputs)

        predicted = np.argmax(ort_outs[0], axis=1)
        total += target_np.shape[0]
        correct += (predicted == target_np).sum().item()

    print(f"Accuracy of the model on the test images: {100 * correct / total} %")


if __name__ == "__main__":
    evaluate()
