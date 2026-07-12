# =========================
# loss curves
# =========================
import matplotlib.pyplot as plt

def plot_history(history, title="Training History"):
    """
    Plots the training and validation loss curves from the training history.

    Args:
        history (dict): A dictionary containing 'train_loss' and 'valid_loss'.
        title (str): The title of the plot.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label='Train Loss')
    plt.plot(history["valid_loss"], label='Valid Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()