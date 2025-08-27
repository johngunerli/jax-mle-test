# JAX ML Project with ONNX Compatibility

This is a template codebase for training machine learning models in JAX, with a focus on ONNX compatibility.

## Project Structure

- `data/`: Directory for storing datasets.
- `notebooks/`: Jupyter notebooks for exploration and visualization.
- `src/`: Source code for the project.
  - `model.py`: Defines the machine learning model using JAX and Flax.
  - `train.py`: Script for training the model.
  - `evaluate.py`: Script for evaluating the trained model.
  - `utils.py`: Utility functions for data loading, preprocessing, etc.
- `requirements.txt`: A list of Python packages required for this project.
- `.gitignore`: Specifies which files and directories to ignore in version control.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model:**
   ```bash
   python -m src.train
   ```

3. **Evaluate the model:**
   ```bash
   python -m src.evaluate
   ```

4. **Visualize with TensorBoard:**
   ```bash
   tensorboard --logdir=runs
   ```

5. **Export to ONNX:**
   The training script will save the model in ONNX format.

## ONNX Compatibility

The model is designed to be compatible with ONNX. The `train.py` script demonstrates how to convert the trained JAX model to the ONNX format.
