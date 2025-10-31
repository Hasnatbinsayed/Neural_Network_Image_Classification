import matplotlib.pyplot as plt
import numpy as np
import json
import os

CIFAR10_LABELS = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def plot_history(history, outdir='outputs/figures'):
    os.makedirs(outdir, exist_ok=True)
    h = history.history
    plt.figure(figsize=(8,4))
    plt.plot(h['loss'], label='train loss')
    plt.plot(h['val_loss'], label='val loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(os.path.join(outdir, 'loss.png'))
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(h['accuracy'], label='train acc')
    plt.plot(h['val_accuracy'], label='val acc')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(outdir, 'accuracy.png'))
    plt.close()

def save_history(history, path='outputs/history.json'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(history.history, f)

def visualize_predictions(model, x_test, y_test, idxs=None, num=10, outdir='outputs/figures'):
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(outdir, exist_ok=True)
    if idxs is None:
        idxs = np.random.choice(range(len(x_test)), num, replace=False)
    preds = model.predict(x_test[idxs])
    pred_classes = preds.argmax(axis=1)
    true_classes = y_test[idxs].squeeze()
    for i, idx in enumerate(idxs):
        plt.figure(figsize=(2.5,2.5))
        plt.imshow((x_test[idx]*255).astype('uint8'))
        plt.axis('off')
        title = f"True: {CIFAR10_LABELS[int(true_classes[i])]} | Pred: {CIFAR10_LABELS[int(pred_classes[i])]}"
        plt.title(title, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'pred_{i}.png'))
        plt.close()
