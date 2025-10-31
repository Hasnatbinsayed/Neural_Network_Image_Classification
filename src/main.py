from train import train
from predict_examples import run_predictions
import argparse
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','predict'], default='train')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    if args.mode == 'train':
        train(epochs=args.epochs, batch_size=args.batch_size)
    elif args.mode == 'predict':
        run_predictions()

if __name__ == '__main__':
    main()
