import logging, os 

import torch
import torch.nn as nn
import torch.optim as optim
import math

from src.modules import LabelSmoothingCrossEntropyLoss
from src.util import calculate_accuracy

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SpeechTransformer')

PAD_TOKEN = 1024
SOS_TOKEN = 1025 
EOS_TOKEN = 1026 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoderModule(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, max_seq_length):
        super(TransformerEncoderModule, self).__init__()
        self.d_model = d_model
        # Initialize the linear layer for processing continuous embeddings
        self.input_linear = nn.Linear(input_dim, d_model, bias=False)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    def forward(self, src, src_mask, src_key_padding_mask):
        # Apply the linear layer to the continuous embeddings
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output

class TransformerDecoderModule(nn.Module):
    def __init__(self, embedding, d_model, nhead, num_decoder_layers, dim_feedforward, dropout, max_seq_length, vocab_size):
        super(TransformerDecoderModule, self).__init__()
        self.embedding = embedding
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_seq_length)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # Note: Embedding weights are shared and initialized in the main model, so they're not re-initialized here.
        self.head.bias.data.zero_()
        self.head.weight.data.uniform_(-initrange, initrange)

    def forward(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_decoder(tgt)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = self.head(output)
        return output

class SpeechTransformerModel(nn.Module):
    """
    A Speech Transformer model for sequence-to-sequence learning tasks.
    
    Attributes:
        shared_embedding (nn.Embedding): A shared embedding layer for both the encoder and decoder.
        encoder (TransformerEncoderModule): The Transformer encoder module.
        decoder (TransformerDecoderModule): The Transformer decoder module.
        optimizer (torch.optim): The optimizer for training the model.
        criterion (nn.CrossEntropyLoss): The loss function.
    
    Parameters:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The dimensionality of the model's embeddings.
        nhead (int): The number of heads in the multiheadattention models.
        num_encoder_layers (int): The number of sub-encoder-layers in the encoder.
        num_decoder_layers (int): The number of sub-decoder-layers in the decoder.
        dim_feedforward (int): The dimension of the feedforward network model.
        dropout (float): The dropout value.
        max_seq_length (int): The maximum sequence length.
        lr (float): The learning rate for the optimizer.
        label_smoothing (float): The `\alpha` in LabelSmoothingCrossEntropyLoss.
        optimizer_type (str): The type of optimizer to use ('sgd' or 'adam').
    """
    def __init__(self, vocab_size, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, max_seq_length, lr, label_smoothing=0.0, optimizer_type="sgd", logger=None):
        super(SpeechTransformerModel, self).__init__()

        self.logger = logger or logging.getLogger('SpeechTransformerModel')
        self.shared_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = TransformerEncoderModule(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, max_seq_length)
        self.decoder = TransformerDecoderModule(self.shared_embedding, d_model, nhead, num_decoder_layers, dim_feedforward, dropout, max_seq_length, vocab_size)
        
        self.init_weights()

        # Optimizer selection
        if optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0)
        elif optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            raise ValueError("Unsupported optimizer type provided. Choose either 'sgd' or 'adam'.")

        # Loss function
        if label_smoothing > 0: 
            self.criterion = LabelSmoothingCrossEntropyLoss(alpha=label_smoothing, ignore_index=PAD_TOKEN)
            self.logger.info(f"Using Label Smoothing CrossEntropyLoss with alpha={label_smoothing}")
        else: # default to standard CE loss 
            self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
            self.logger.info("Using standard CrossEntropyLoss")
        
        # Print parameters after initialization
        self.print_parameters()

    def print_parameters(self):
        total_params = 0
        for name, param in self.named_parameters():
            self.logger.info(f"{name}, shape: {param.size()}")
            total_params += param.numel()
        self.logger.info(f"Total parameters: {total_params}")

    def init_weights(self):
        initrange = 0.1
        self.shared_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.init_weights()

    def encoder_masked_avg_pooling(self, encoder_output, src_key_padding_mask):
        """
        Performs masked average pooling on the encoder's output.

        This method computes the mean of each feature across the sequence length, ignoring the elements
        marked as padding in the `src_key_padding_mask`. This ensures that the pooled representation
        does not include the influence of padding tokens, providing a more accurate summary of the
        actual sequence content. The operation is performed per batch and feature dimension.

        Parameters:
        - encoder_output (torch.Tensor): The output from the encoder with shape [seq_len, batch_size, hidden_dim],
          where `seq_len` is the length of the sequence, `batch_size` is the number of samples in the batch, and
          `hidden_dim` is the dimensionality of the encoder's output features.
        - src_key_padding_mask (torch.Tensor): A boolean tensor indicating which elements of the sequence are padding tokens,
          with shape [batch_size, seq_len]. `True` values indicate padding tokens, and `False` values indicate actual sequence elements.

        Returns:
        - torch.Tensor: The masked average pooled representation of the encoder's output, with shape [batch_size, hidden_dim].
          This tensor provides a single vector representation per batch item, summarizing the non-padding elements of the input sequence.
        """
        # Permute encoder_output to [batch_size, hidden_dim, seq_len]
        encoder_output = encoder_output.permute(1, 2, 0)
        
        # Invert the mask: True for valid, False for pad
        mask = ~src_key_padding_mask
        
        # Expand mask to match encoder_output's shape
        mask_expanded = mask.unsqueeze(1).expand(-1, encoder_output.size(1), -1).to(encoder_output.dtype)
        
        # Apply mask, sum over seq_len, and compute the sum of valid (non-pad) positions
        sum_pool = (encoder_output * mask_expanded).sum(dim=2)
        valid_counts = mask_expanded.sum(dim=2)
        
        # Avoid division by zero for sequences that are fully padded
        valid_counts = valid_counts.masked_fill(valid_counts == 0, 1)
        
        # Compute masked average
        masked_avg = sum_pool / valid_counts
        
        return masked_avg

    def forward(self, src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask):
        """
        Performs the forward pass of the model.
        
        Parameters:
            src (Tensor): The source sequence.
            tgt (Tensor): The target sequence.
            src_mask (Tensor): The source mask.
            tgt_mask (Tensor): The target mask.
            src_key_padding_mask (Tensor): The source key padding mask.
            tgt_key_padding_mask (Tensor): The target key padding mask.
            memory_key_padding_mask (Tensor): The memory key padding mask.
        
        Returns:
            Tensor: The output of the decoder.
        """
        encoder_output = self.encoder(src, src_mask, src_key_padding_mask) # [seq_len, batch_size, hidden_dim]
        # Use the masked_avg_pooling method for pooling encoder outputs to a single representation
        pooled_output = self.encoder_masked_avg_pooling(encoder_output, src_key_padding_mask) # [batch_size, hidden_dim]

        context_size = tgt.size(1) // encoder_output.size(1)
        pooled_output_expanded = pooled_output.repeat_interleave(context_size, dim=0) # repeat by context size
        memory_key_padding_mask = memory_key_padding_mask.repeat_interleave(context_size, dim=0) # repeat by context size
        #import pdb; pdb.set_trace()
        pooled_output_expanded = pooled_output_expanded.unsqueeze(0)
        output = self.decoder(tgt, pooled_output_expanded, tgt_mask, tgt_key_padding_mask, None)
        
        return output

    def encoder_forward_pass(self, src, src_mask, src_key_padding_mask):
        """
        Performs a forward pass through the encoder only.
        
        Parameters:
            src (Tensor): The source sequence.
            src_mask (Tensor): The source mask.
            src_key_padding_mask (Tensor): The source key padding mask.
        
        Returns:
            Tensor: The pooled encoder output.
        """
        encoder_output = self.encoder(src, src_mask, src_key_padding_mask)
        pooled_output = self.encoder_masked_avg_pooling(encoder_output, src_key_padding_mask) # [batch_size, hidden_dim]
        return pooled_output

    def train_step(self, src, tgt_input, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask):
        """
        Performs a single training step, including forward pass, loss calculation, backpropagation, and optimization.

        Parameters:
        - src (Tensor): Source sequences tensor with shape (seq_len, batch_size).
        - tgt_input (Tensor): Input target sequences tensor with shape (seq_len, batch_size).
        - tgt (Tensor): Original target sequences tensor with shape (seq_len, batch_size).
        - src_mask (Tensor): Source sequence mask tensor.
        - tgt_mask (Tensor): Target sequence mask tensor.
        - src_key_padding_mask (Tensor): Source key padding mask tensor.
        - tgt_key_padding_mask (Tensor): Target key padding mask tensor.
        - memory_key_padding_mask (Tensor): Memory key padding mask tensor for attention mechanisms.

        The method computes the model's output, reshapes it for loss calculation, computes the loss, and performs a backward pass to update the model's parameters. 
        Gradient clipping is applied to prevent exploding gradients. Finally, it calculates and returns the loss and accuracy of the predictions.

        Returns:
        - A tuple containing:
            - The loss value as a float.
            - A tuple with the accuracy of the predictions and the number of correct predictions.
        """
        self.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.forward(src, tgt_input, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)

        # Reshape for loss calculation
        output_reshaped = output.reshape(-1, output.size(-1))
        tgt_reshaped = tgt.reshape(-1)

        total_loss = self.criterion(output_reshaped, tgt_reshaped)

        # Backpropagate the total loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Calculate accuracy
        with torch.no_grad():
            accuracy, predictions, ground_truths = calculate_accuracy(output_reshaped, tgt_reshaped, PAD_TOKEN)
        
        return total_loss.item(), (accuracy, predictions, ground_truths)

    def eval_step(self, src, tgt_input, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask):
        """
        Evaluates the model on a given batch of data, calculating loss and accuracy without performing any backpropagation.

        Parameters:
        - src (Tensor): Source sequences tensor with shape (seq_len, batch_size).
        - tgt_input (Tensor): Input target sequences tensor with shape (seq_len, batch_size).
        - tgt (Tensor): Original target sequences tensor with shape (seq_len, batch_size).
        - src_mask (Tensor): Source sequence mask tensor.
        - tgt_mask (Tensor): Target sequence mask tensor.
        - src_key_padding_mask (Tensor): Source key padding mask tensor.
        - tgt_key_padding_mask (Tensor): Target key padding mask tensor.
        - memory_key_padding_mask (Tensor): Memory key padding mask tensor for attention mechanisms.

        This method is similar to `train_step` but is used for model evaluation. It disables gradient calculation, computes the output, loss, and accuracy, and returns them.

        Returns:
        - A tuple containing:
            - The loss value as a float.
            - A tuple with the accuracy of the predictions and the number of correct predictions.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(src, tgt_input, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)

            # Reshape for loss calculation
            output_reshaped = output.reshape(-1, output.size(-1))
            tgt_reshaped = tgt.reshape(-1)

            total_loss = self.criterion(output_reshaped, tgt_reshaped)

            # Calculate accuracy
            accuracy, predictions, ground_truths = calculate_accuracy(output_reshaped, tgt_reshaped, PAD_TOKEN)

        return total_loss.item(), (accuracy, predictions, ground_truths)

    def inference_step(self, src, src_mask, src_key_padding_mask):
        """
        Generates the encoder's pooled output representation for a given source sequence without performing any decoding.

        Parameters:
        - src (Tensor): Source sequences tensor with shape (seq_len, batch_size).
        - src_mask (Tensor): Source sequence mask tensor.
        - src_key_padding_mask (Tensor): Source key padding mask tensor.

        This method is intended for scenarios where only the encoder's output is required, for instance, when using the model for feature extraction. 
        It switches the model to evaluation mode, computes the encoder's output, applies pooling, and returns the pooled representation.

        Returns:
        - Tensor: The pooled output of the encoder with shape (batch_size, hidden_dim).
        """
        self.eval()
        with torch.no_grad():
            pooled_output = self.encoder_forward_pass(src, src_mask, src_key_padding_mask)
        return pooled_output

    def greedy_decode_step(self, src, src_mask, src_key_padding_mask, max_length=50):
        """
        Greedy decoding implementation for the model, handling variable output length with EOS_TOKEN detection.

        Parameters:
        - src (Tensor): The source sequence.
        - src_mask (Tensor): The source mask.
        - src_key_padding_mask (Tensor): The source key padding mask.

        Returns:
        - Tensor: The generated sequence.
        """
        max_decoding_steps = max_length - 2

        self.eval()
        with torch.no_grad():
            # Prepare pooled_output_expanded as per the updated model's forward pass
            pooled_output = self.encoder_forward_pass(src, src_mask, src_key_padding_mask)
            pooled_output_expanded = pooled_output.unsqueeze(0)  # Ensure correct shape for decoder

            # Initialize the target tensor with SOS_TOKEN at the first position
            tgt = torch.full((1, src.shape[1]), SOS_TOKEN, dtype=torch.long, device=src.device)

            generated = tgt

            # Continue decoding until EOS_TOKEN is generated
            for _ in range(max_decoding_steps):

                # Call the decoder with the current state of tgt
                # No tgt_mask for greedy decoding as each step only looks at previously generated tokens
                output = self.decoder(tgt, pooled_output_expanded, None, None, None)
            
                
                # Select the last step's output for next token prediction
                next_token_logits = output[-1, :, :]  # Get logits of the last token
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # Predict the next token
                next_token = next_token.transpose(0, 1)

                # Break the loop if EOS_TOKEN is generated
                if (next_token == EOS_TOKEN).all():
                    break

                # Append the predicted token to the generated sequence
                generated = torch.cat([generated, next_token], dim=0)

                # Update tgt with the newly generated token for the next iteration
                tgt = torch.cat([tgt, next_token], dim=0)
                
        return generated

    def save_model(self, epoch, file_save_path):
        if not os.path.exists(os.path.dirname(file_save_path)):
            os.makedirs(os.path.dirname(file_save_path))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, file_save_path)

    def load_model(self, file_load_path):
        checkpoint = torch.load(file_load_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        self.logger.info(f"Model and optimizer loaded. Resuming from epoch {epoch}.")
        return epoch

