import os
import numpy as np
from sklearn.metrics import precision_score, recall_score,f1_score
import glob

def load_labels_from_txt(path):
    labels = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            labels.append(class_id)
    return labels


def evaluate_predictions(pred_dir, gt_dir):
    y_true = []
    y_pred = []

    for label_file in glob.glob(os.path.join(gt_dir, '*.txt')):
        file_name = os.path.basename(label_file)
        gt_labels = load_labels_from_txt(label_file)

        pred_file = os.path.join(pred_dir, file_name)
        if os.path.exists(pred_file):
            pred_labels = load_labels_from_txt(pred_file)
        else:
            pred_labels = []

        # Match the number of predictions to GT labels for scoring simplicity
        min_len = min(len(gt_labels), len(pred_labels))
        y_true.extend(gt_labels[:min_len])
        y_pred.extend(pred_labels[:min_len])

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")


if __name__ == '__main__':
    # Update these paths as needed
    pred_path = 'runs/detect/predict/labels'   # YOLO output labels path
    gt_path = 'datasets/acw2_images/yolo/val/labels'  # Ground truth labels path

    evaluate_predictions(pred_path, gt_path)
