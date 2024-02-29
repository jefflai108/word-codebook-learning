import time
import logging, os 
import argparse

import numpy as np
import torch
from scipy.io import savemat

from src.model import SpeechTransformerModel
from src.data import get_eval_loader, get_inference_loader
from src.util import plot_pred_label_distribution

def configure_logging(save_dir):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(save_dir, 'train.log'), filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    console.setFormatter(formatter)

    # Get the root logger
    logger = logging.getLogger('')
    # Add the console handler to it
    logger.addHandler(console)

    return logger 

def parse_args():
    parser = argparse.ArgumentParser(description="Word codebook learning.")

	# Paths
    parser.add_argument("--train_file_path", type=str, required=True, help="Path to the train file.")
    parser.add_argument("--test_file_path", type=str, required=True, help="Path to the test file.")
    parser.add_argument("--dev_file_path", type=str, required=True, help="Path to the dev file.")
    parser.add_argument("--load_model_path", type=str, required=True, help="Path to the saved model.")
    parser.add_argument("--word_seg_file_path", type=str, required=True, help="Path to the word segmentation file.")
    parser.add_argument('--save_dir', type=str, default='exp/debug/', help='Directory to save model checkpoints and logs')

    # Batch size and context size
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")
    parser.add_argument("--segment_context_size", type=int, default=3, help="Left / right segment context size for the dataset.")

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=100, help="Vocabulary size.")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--nhead", type=int, default=8, help="Number of heads in the multiheadattention models.")
    parser.add_argument("--num_encoder_layers", type=int, default=3, help="Number of encoder layers in the transformer.")
    parser.add_argument("--num_decoder_layers", type=int, default=3, help="Number of decoder layers in the transformer.")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="Dimension of the feedforward network model.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout value.")
    parser.add_argument("--max_seq_length", type=int, default=120, help="Maximum sequence length.")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing parameter between 0 and 1. 0 disables label smoothing.")

    # Training specifics
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--log_interval', type=int, default=100, help='Log loss every this many intervals')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer type.")

    return parser.parse_args()

def inference(model, num_classes, inference_dataloader, logger, device): 
    total_predictions_per_class = torch.zeros(num_classes, dtype=torch.long)
    total_targets_per_class = torch.zeros(num_classes, dtype=torch.long)

    # Initialize a dictionary to hold all encoder representations
    all_encoder_representations = {}

    model.eval()
    with torch.no_grad():
        for uttid, X, src_mask, src_key_padding_mask in inference_dataloader:
            assert len(uttid) == 1

            # Move tensors to the right device
            X, src_key_padding_mask = X.to(device), src_key_padding_mask.to(device)

            # make the input [sequence_length, batch_size] instead 
            X = X.transpose(0, 1) 
            
            #import pdb; pdb.set_trace()
            encoder_repre = model.inference_step(X, src_mask, src_key_padding_mask)
            
            # Convert encoder representations to NumPy and store with utterance key
            all_encoder_representations[uttid[0]] = encoder_repre.cpu().numpy()

    return all_encoder_representations


def greedy_decode(): 
    pass
    # do actual greedy decoding? 
    # or not? 
    # given a test context, load a target model, and calcualte the pred acc 

        #loss, (acc, preds, targets) = model.inference_step(X, src_mask, src_key_padding_mask)

        #    for i in range(num_classes):
        #        # Update ground truths for class i
        #        total_targets_per_class[i] += (targets == i).sum().item()
        #        # Update total predictions for class i
        #        total_predictions_per_class[i] += (preds == i).sum().item()
       
        ## Calculate overall accuracy directly
        #correct_predictions = torch.min(total_predictions_per_class, total_targets_per_class).sum().item()  # Account for potential mismatches
        #total_predictions = total_predictions_per_class.sum().item()
        #overall_accuracy = correct_predictions / total_predictions	

        #import pdb; pdb.set_trace()

        #total_predictions_per_class = [(label, x) for label, x in enumerate(total_predictions_per_class.numpy())]
        #total_targets_per_class = [(label, x) for label, x in enumerate(total_targets_per_class.numpy())]
		#

        #import pdb; pdb.set_trace()
        #plot_pred_label_distribution(total_predictions_per_class, total_targets_per_class)
        #
        ## Sort total_predictions_per_class and get the sorted indices (class labels)
        #sorted_predictions, sorted_prediction_classes = torch.sort(total_predictions_per_class, descending=True)
        #sorted_targets, sorted_targets_classes = torch.sort(total_targets_per_class, descending=True)

        ## Create a list of (class, number of predictions) tuples
        #sorted_prediction_distribution = [(int(cls), predictions.item()) for cls, predictions in zip(sorted_prediction_classes, sorted_predictions)]
        #sorted_ground_truth_distribution = [(int(cls), targets.item()) for cls, targets in zip(sorted_targets_classes, sorted_targets)]
        #print(sorted_prediction_distribution)
        #print(sorted_ground_truth_distribution)

def run_inference_and_save(model, vocab_size, data_loader, logger, device, save_dir, file_name):
    encoder_representation = inference(model, vocab_size, data_loader, logger, device)
    save_path = os.path.join(save_dir, file_name)
    savemat(save_path, encoder_representation)

def main():
    args = parse_args()
    
    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Configure logging to file and console
    logger = configure_logging(args.save_dir)

    # Setup device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize the model
    model = SpeechTransformerModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
        lr=args.learning_rate,
        label_smoothing=args.label_smoothing, 
        optimizer_type=args.optimizer_type, 
        logger=logger, 
    )

    model = model.to(device)
    model.load_model(args.load_model_path)

    # Setup dataloaders 
    train_data_loader = get_inference_loader(args.train_file_path, args.word_seg_file_path, args.segment_context_size, batch_size=1, shuffle=False, num_workers=2, max_seq_len=args.max_seq_length)
    test_data_loader = get_inference_loader(args.test_file_path, args.word_seg_file_path, args.segment_context_size, batch_size=1, shuffle=False, num_workers=2, max_seq_len=args.max_seq_length)
    dev_data_loader = get_inference_loader(args.dev_file_path, args.word_seg_file_path, args.segment_context_size, batch_size=1, shuffle=False, num_workers=2, max_seq_len=args.max_seq_length)

    # Run inference 
    run_inference_and_save(model, args.vocab_size, test_data_loader, logger, device, args.save_dir, "test_encoder_segment_representations.mat")
    run_inference_and_save(model, args.vocab_size, dev_data_loader, logger, device, args.save_dir, "dev_encoder_segment_representations.mat")

if __name__ == "__main__":  
    main()

