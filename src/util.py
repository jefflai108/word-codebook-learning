import torch 
import matplotlib.pyplot as plt

def calculate_accuracy(output_reshaped, tgt_reshaped, pad_token=1024):
    """
    Calculates the prediction accuracy by comparing the predicted labels against the true labels, 
    excluding any instances where the target is a PAD_TOKEN.
    
    Parameters:
    - output_reshaped (Tensor): The model's output logits reshaped to (N, num_classes), where N is 
      the total number of samples in the batch, and num_classes is the number of possible labels.
    - tgt_reshaped (Tensor): The ground truth labels reshaped to a vector of length N.
    
    Returns:
    - accuracy (float): The proportion of correct predictions over the total number of non-pad predictions.
    """
    pred = torch.argmax(output_reshaped, dim=1)  # Get the index of the max log-probability.
    valid_indices = tgt_reshaped != pad_token  # Identify non-PAD_TOKEN elements.
    
    # Filter out PAD_TOKEN predictions and targets.
    valid_preds = pred[valid_indices]
    valid_targets = tgt_reshaped[valid_indices]
    
    # Count how many predictions match the target labels.
    correct = valid_preds.eq(valid_targets).sum().item()
    
    # Calculate the total number of non-pad labels.
    total = valid_targets.size(0)
    
    # Calculate accuracy, avoiding division by zero.
    accuracy = correct / total if total > 0 else 0

    return accuracy, valid_preds, valid_targets

def plot_pred_label_distribution(predictions, ground_truths): 
    # Sort the predictions and ground truths by label for consistency
    predictions_sorted = sorted(predictions, key=lambda x: x[0])
    ground_truths_sorted = sorted(ground_truths, key=lambda x: x[0])

    # Unpack the labels and counts
    pred_labels, pred_counts = zip(*predictions_sorted)
    gt_labels, gt_counts = zip(*ground_truths_sorted)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot both distributions
    ax.bar(pred_labels, pred_counts, width=0.4, align='center', alpha=0.6, label='Predictions')
    ax.bar(gt_labels, gt_counts, width=0.4, align='edge', alpha=0.6, label='Ground Truths')

    # Adding labels and title
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    ax.set_title('Comparison of Prediction and Ground Truth Distributions')
    ax.legend()

    # Show plot
    plt.xticks(rotation=90) # Rotate labels to avoid overlap
    plt.show()
    plt.savefig('shit.png')

