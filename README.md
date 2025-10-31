# Neural Networks and Their Application to Image Classification

## Project Overview
This project demonstrates a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset using TensorFlow and Keras.

## Setup Instructions
1. Open this project folder in **PyCharm**.
2. Create a virtual environment using **Python 3.8**.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the training script:
   ```bash
   python src/main.py --mode train --epochs 10
   ```
5. To visualize predictions:
   ```bash
   python src/main.py --mode predict
   ```

## Output Files
- `outputs/model.h5` — trained model
- `outputs/history.json` — training history
- `outputs/figures/` — accuracy and loss plots, sample predictions

The CIFAR-10 dataset will download automatically when you run the script.
