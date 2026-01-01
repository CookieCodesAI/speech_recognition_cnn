from sklearn.metrics import roc_curve, ConfusionMatrixDisplay, confusion_matrix, accuracy_score, auc
from sklearn.preprocessing import label_binarize
from preprocessing import get_labels
import matplotlib.pyplot as plt
import numpy as np

def evaluate(model, test_ds, info):
    labels = get_labels(info)
    num_classes = len(labels)
    print("Labels: ",  labels)

    loss, accuracy = model.evaluate(test_ds, batch_size=32)
    print(f"Loss: {loss}. Accuracy: {accuracy}.")

    y_true = np.concatenate([y for _, y in test_ds], axis=0)
    y_score = model.predict(test_ds)
    y_pred = np.argmax(y_score, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap="Purples")
    plt.show()

    y_true_bin = label_binarize(y_true, classes = list(range(num_classes)))
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:,i], y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label = f"{labels[i]} (AUC = {roc_auc[i]:.2f})")
    
    plt.plot([0,1], [0,1], "k--")
    plt.title("ROC Curves per Class")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


