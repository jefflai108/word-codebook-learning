import time
import logging, os 
import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.model import SpeechTransformerModel
from src.data import get_train_loader, get_eval_loader

PAD_TOKEN = 1024
SOS_TOKEN = 1025 
EOS_TOKEN = 1026 

def configure_logging(save_dir):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(save_dir, f'training.log'), filemode='a')
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
    parser.add_argument("--train_token_file_path", type=str, required=True, help="Path to the train token file.")
    parser.add_argument("--dev_token_file_path", type=str, required=True, help="Path to the dev token file.")
    parser.add_argument("--train_embed_file_paths", type=str, required=True, help="Comma-separated paths to the train embed files.")
    parser.add_argument("--dev_embed_file_paths", type=str, required=True, help="Comma-separated paths to the dev embed files.")
    parser.add_argument("--word_seg_file_path", type=str, required=True, help="Path to the word segmentation file.")
    parser.add_argument('--save_dir', type=str, default='exp/debug/', help='Directory to save model checkpoints and logs')

    # Batch size and context size
    parser.add_argument("--num_layer_repre", type=int, default=5, help="Number of input layer repre (concatenated).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=100, help="Vocabulary size.")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--repre_dim", type=int, default=1024, help="Feature dimension of the representations.")
    parser.add_argument("--nhead", type=int, default=8, help="Number of heads in the multiheadattention models.")
    parser.add_argument("--num_encoder_layers", type=int, default=3, help="Number of encoder layers in the transformer.")
    parser.add_argument("--num_decoder_layers", type=int, default=3, help="Number of decoder layers in the transformer.")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="Dimension of the feedforward network model.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout value.")
    parser.add_argument("--max_seq_length", type=int, default=120, help="Maximum sequence length.")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing parameter between 0 and 1. 0 disables label smoothing.")
    parser.add_argument("--model_activation", type=str, default="relu", choices=["relu", "gelu"], help="activation functions used in transformer modules.")

    # Training specifics
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--log_interval', type=int, default=100, help='Log loss every this many intervals')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer type.")
    parser.add_argument('--gradient_acc_steps', type=int, default=2, help='number of training steps accumulated')

    return parser.parse_args()

def adjust_learning_rate(optimizer, step, total_steps, peak_lr, end_lr=1e-6):
    warmup_steps = int(total_steps * 0.2)  # 20% of total steps for warm-up
    decay_steps = int(total_steps * 0.8)  # 80% of total steps for decay

    if step < warmup_steps:
        lr = peak_lr * step / warmup_steps
    elif step < warmup_steps + decay_steps:
        step_into_decay = step - warmup_steps
        lr = peak_lr * (end_lr / peak_lr) ** (step_into_decay / decay_steps)
    else:
        lr = end_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model, train_dataloader, dev_dataloader, current_epoch, epochs, logger, tb_writer, log_interval, peak_lr, gradient_acc_steps, save_dir, device):
    total_steps = epochs * len(train_dataloader)
    current_step = current_epoch * len(train_dataloader)
    best_loss, best_acc = float('inf'), 0.0

    for epoch in range(current_epoch, epochs):
        def _train(current_step):
            model.train()
            total_loss, total_acc = 0, 0
            log_loss, log_acc = 0, 0 
            for batch_idx, (uttids, X, Y, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask) in enumerate(train_dataloader):
                current_step += 1
                adjust_learning_rate(model.optimizer, current_step, total_steps, peak_lr)

                # Move tensors to the right device
                X, Y = X.to(device), Y.to(device)
                tgt_mask = tgt_mask.to(device)
                src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = src_key_padding_mask.to(device), tgt_key_padding_mask.to(device), memory_key_padding_mask.to(device)

                # make the input [sequence_length, batch_size] instead 
                X = X.transpose(0, 1) 
                Y = Y.transpose(0, 1)

                # shift targets for teacher forcing
                Y_input = torch.cat([torch.full((1, Y.shape[1]), SOS_TOKEN, dtype=Y.dtype, device=Y.device), Y[:-1]], dim=0)

                # Forward pass
                loss, (acc, preds, targets) = model.train_step(X, Y_input, Y, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask, current_step, gradient_acc_steps)
                log_loss += loss
                log_acc += acc 

                # Log loss every log_interval steps
                if (batch_idx + 1) % log_interval == 0:
                    total_loss += log_loss
                    total_acc += log_acc
                    log_avg_loss = log_loss / log_interval
                    log_avg_acc  = log_acc / log_interval
                    current_lr = model.optimizer.param_groups[0]['lr']
                    tb_writer.add_scalar('Loss/train', log_avg_loss, epoch * len(train_dataloader) + batch_idx)
                    tb_writer.add_scalar('Accuracy/train', log_avg_acc, epoch * len(train_dataloader) + batch_idx)
                    tb_writer.add_scalar('Learning_Rate', current_lr, epoch * len(train_dataloader) + batch_idx)
                    logging.info(f"Epoch: {epoch}, Step: {batch_idx+1}, Averaged Loss: {log_avg_loss:.4f}, Averaged Acc: {log_avg_acc:.4f}, LR: {current_lr:.5f}")
                    log_loss, log_acc = 0, 0

                # free-up GPU mem 
                del X, Y, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask

            avg_loss = total_loss / len(train_dataloader)
            avg_acc = total_acc / len(train_dataloader) 
            logger.info(f'===> Epoch {epoch} Complete: Avg. Loss: {avg_loss:.4f}. Acc: {avg_acc:.4f}')
            return avg_loss, current_step

        def _val():
            model.eval()
            total_loss, total_acc = 0, 0
            with torch.no_grad():
                for uttids, X, Y, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask in dev_dataloader:
                    # Move tensors to the right device
                    X, Y = X.to(device), Y.to(device)
                    tgt_mask = tgt_mask.to(device)
                    src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = src_key_padding_mask.to(device), tgt_key_padding_mask.to(device), memory_key_padding_mask.to(device)

                    # make the input [sequence_length, batch_size] instead 
                    X = X.transpose(0, 1) 
                    Y = Y.transpose(0, 1)

                    # shift targets for teacher forcing
                    Y_input = torch.cat([torch.full((1, Y.shape[1]), SOS_TOKEN, dtype=Y.dtype, device=Y.device), Y[:-1]], dim=0)

                    loss, (acc, preds, targets) = model.eval_step(X, Y_input, Y, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
                    total_loss += loss
                    total_acc += acc
            
            avg_loss = total_loss / len(dev_dataloader)
            avg_acc = total_acc / len(dev_dataloader)
            tb_writer.add_scalar('Loss/val', avg_loss, epoch)
            tb_writer.add_scalar('Accuracy/val', avg_acc, epoch)
            logger.info(f'===> Validation set: Average loss: {avg_loss:.4f}. Average acc: {avg_acc:.4f}')
            return avg_loss, avg_acc 

        train_loss, current_step = _train(current_step)
        val_loss, val_acc = _val()
    
        model_save_path = os.path.join(save_dir, f'model_epoch_{epoch}_loss_{val_loss:.4f}.pth')
        model.save_model(epoch, model_save_path)
        logger.info(f'Model saved to {model_save_path}') 
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_save_path = os.path.join(save_dir, f'best_loss_model.pth')
            model.save_model(epoch, best_model_save_path)
        if val_acc > best_acc: 
            best_acc = val_acc 
            best_model_save_path = os.path.join(save_dir, f'best_acc_model.pth')
            model.save_model(epoch, best_model_save_path)

def main():
    args = parse_args()
    
    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Configure logging to file and console
    logger = configure_logging(args.save_dir)
    tb_writer = SummaryWriter(args.save_dir)

    # Setup device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    train_embed_file_paths = args.train_embed_file_paths.split(';') 
    dev_embed_file_paths = args.dev_embed_file_paths.split(';')

    # Initialize the model
    model = SpeechTransformerModel(
        vocab_size=args.vocab_size,
        input_dim=args.repre_dim,
        num_layer_repre=args.num_layer_repre, 
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
        lr=args.learning_rate,
        activation=args.model_activation,
        label_smoothing=args.label_smoothing, 
        optimizer_type=args.optimizer_type, 
        logger=logger, 
    )

    model = model.to(device)
    if os.path.exists(os.path.join(args.save_dir, f'best_loss_model.pth')):
        current_epoch = model.load_model(os.path.join(args.save_dir, f'best_loss_model.pth'))
    else: current_epoch = 0 

    # Setup dataloaders 
    train_data_loader = get_train_loader(args.train_token_file_path, train_embed_file_paths, args.word_seg_file_path, args.batch_size, shuffle=True, num_workers=1, max_seq_len=args.max_seq_length)
    dev_data_loader = get_eval_loader(args.dev_token_file_path, dev_embed_file_paths, args.word_seg_file_path, args.batch_size, shuffle=False, num_workers=1, max_seq_len=args.max_seq_length)

    # Run the training
    train(model, train_data_loader, dev_data_loader, current_epoch, args.epochs, logger, tb_writer, args.log_interval, args.learning_rate, args.gradient_acc_steps, args.save_dir, device)
    tb_writer.close()

if __name__ == "__main__":  
    main()

