import torch 

def calculate_accuracy(output_reshaped, tgt_reshaped):
    """
    Calculates the prediction accuracy by comparing the predicted labels against the true labels.

    Parameters:
    - output_reshaped (Tensor): The model's output logits reshaped to (N, num_classes), where N is the total number of samples in the batch, and num_classes is the number of possible labels.
    - tgt_reshaped (Tensor): The ground truth labels reshaped to a vector of length N.

    Returns:
    - accuracy (float): The proportion of correct predictions over the total number of predictions.
    """
    pred = torch.argmax(output_reshaped, dim=1)  # Get the index of the max log-probability
    correct = pred.eq(tgt_reshaped)  # Count how many predictions match the target labels
    total = tgt_reshaped.size(0)  # Total number of labels
    accuracy = correct.sum().item() / total  # Calculate accuracy

    return accuracy, pred, tgt_reshaped

