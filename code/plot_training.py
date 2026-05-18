import os
import argparse
import pickle
import matplotlib.pyplot as plt
import pandas as pd

def plot_training_curves(history_path, output_dir):
    """
    Plot training and validation accuracy and loss from a saved history object.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(history_path):
        print(f"Warning: History file {history_path} does not exist.")
        print("Fallback Strategy: If the model was trained elsewhere (e.g., Google Colab) and the history object was not saved,")
        print("you have two options:")
        print("1. Re-run training for a few epochs locally to generate a new history object.")
        print("2. If you have CSV logs of the training (e.g., from CSVLogger), you can parse those instead of a .pkl file.")
        print("Creating placeholder mock graphs for demonstration purposes...")
        create_mock_curves(output_dir)
        return

    print(f"Loading history from {history_path}...")
    try:
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
    except Exception as e:
        # Fallback if it's a csv
        try:
            history_df = pd.read_csv(history_path)
            history = history_df.to_dict(orient='list')
        except:
            print(f"Error loading history: {e}")
            return

    # Assuming history is a dict with 'accuracy', 'val_accuracy', 'loss', 'val_loss'
    epochs = range(1, len(history.get('accuracy', history.get('acc', []))) + 1)
    
    # 1. Accuracy Curve
    plt.figure(figsize=(10, 6))
    if 'accuracy' in history:
        plt.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
    elif 'acc' in history:
        plt.plot(epochs, history['acc'], 'b-', label='Training Accuracy')
        
    if 'val_accuracy' in history:
        plt.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
    elif 'val_acc' in history:
        plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(output_dir, 'accuracy_curve.png')
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"Saved accuracy curve to {acc_path}")

    # 2. Loss Curve
    plt.figure(figsize=(10, 6))
    if 'loss' in history:
        plt.plot(epochs, history['loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"Saved loss curve to {loss_path}")

def create_mock_curves(output_dir):
    """Fallback function to create mock curves if no history is found."""
    import numpy as np
    epochs = np.arange(1, 21)
    train_acc = 1 - np.exp(-epochs/5)
    val_acc = train_acc * 0.95 + np.random.normal(0, 0.02, len(epochs))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy (Mock)')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy (Mock)')
    plt.title('Training and Validation Accuracy (Mock Data)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'), dpi=300)
    plt.close()
    
    train_loss = np.exp(-epochs/5)
    val_loss = train_loss * 1.1 + np.random.normal(0, 0.05, len(epochs))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss (Mock)')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss (Mock)')
    plt.title('Training and Validation Loss (Mock Data)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300)
    plt.close()
    print(f"Saved mock curves to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and validation curves")
    parser.add_argument('--history', type=str, default='../models/history.pkl', help="Path to saved history object (.pkl or .csv)")
    parser.add_argument('--output', type=str, default='../outputs', help="Output directory")
    
    args = parser.parse_args()
    plot_training_curves(args.history, args.output)
