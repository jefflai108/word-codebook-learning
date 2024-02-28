import logging, os 

import torch
import torch.nn as nn
import torch.optim as optim
import math

from src.util import calculate_accuracy

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SpeechTransformer')

PAD_TOKEN = 1024

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
    def __init__(self, embedding, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, max_seq_length):
        super(TransformerEncoderModule, self).__init__()
        self.d_model = d_model
        self.embedding = embedding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    def forward(self, src, src_mask, src_key_padding_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
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
        pooling (nn.AdaptiveAvgPool1d): An adaptive average pooling layer to pool encoder outputs.
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
        optimizer_type (str): The type of optimizer to use ('sgd' or 'adam').
    """
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, max_seq_length, lr, optimizer_type="sgd", logger=None):
        super(SpeechTransformerModel, self).__init__()

        self.logger = logger or logging.getLogger('SpeechTransformerModel')
        self.shared_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = TransformerEncoderModule(self.shared_embedding, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, max_seq_length)
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Pooling to get single representation
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
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)  

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
        # Pool encoder outputs to a single representation
        pooled_output = self.pooling(encoder_output.permute(1, 2, 0)).squeeze(-1) # [batch_size, hidden_dim] 

        context_size = tgt.size(1) // encoder_output.size(1)
        pooled_output_expanded = pooled_output.repeat_interleave(context_size, dim=0) # repeat by context size
        memory_key_padding_mask = memory_key_padding_mask.repeat_interleave(context_size, dim=0) # repeat by context size
        pooled_output_expanded = pooled_output_expanded.repeat(src.size(0), 1, 1) # repeat by src_seq_len
        output = self.decoder(tgt, pooled_output_expanded, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
        
        #B* T = tgt.shape
        ## Prepare an empty tensor to hold decoder outputs for each context
        #all_decoder_outputs = torch.zeros(B, C, T, device=tgt.device)  # Ensure this matches your target vocab size and device
        #
        ## Iterate over each context in Y
        #for i in range(C):
        #    current_tgt = tgt[:, i, :]  # Selecting the i-th context
        #    # Expand pooled output to match current_tgt sequence length
        #    pooled_output_expanded = pooled_output.unsqueeze(1).repeat(1, current_tgt.size(1), 1)
        #    current_output = self.decoder(current_tgt, pooled_output_expanded, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
        #    all_decoder_outputs[:, i, :] = current_output
        
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
        pooled_output = self.pooling(encoder_output.permute(1, 2, 0)).squeeze(-1) # [batch_size, hidden_dim]
        return pooled_output

    def train_step(self, src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask):
        """
        Performs a single training step, including forward pass, loss calculation, backpropagation, and optimization.

        Parameters:
        - src (Tensor): Source sequences tensor with shape (seq_len, batch_size).
        - tgt (Tensor): Target sequences tensor with shape (seq_len, batch_size).
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
        
        # Output shape should now consider the context size: (B, C, T, vocab_size)
        output = self.forward(src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)

        # Reshape for loss calculation
        output_reshaped = output.reshape(-1, output.size(-1))
        tgt_reshaped = tgt.reshape(-1)

        total_loss = self.criterion(output_reshaped, tgt_reshaped)

        ## Accumulate losses for each context
        #total_loss = 0
        #for i in range(tgt.size(1)):  # Iterate over the context dimension
        #    current_tgt = tgt[:, i, :].contiguous().view(-1)  # Flatten the i-th context targets
        #    current_output = output[:, i, :].contiguous().view(-1, output.size(-1))  # Flatten the i-th context outputs
        #    loss = self.criterion(current_output, current_tgt)
        #    total_loss += loss
        
        # Backpropagate the total loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Calculate accuracy
        with torch.no_grad():
            accuracy, predictions, ground_truths = calculate_accuracy(output_reshaped, tgt_reshaped)
        
        return total_loss.item(), (accuracy, predictions, ground_truths)

    def eval_step(self, src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask):
        """
        Evaluates the model on a given batch of data, calculating loss and accuracy without performing any backpropagation.

        Parameters:
        - src (Tensor): Source sequences tensor with shape (seq_len, batch_size).
        - tgt (Tensor): Target sequences tensor with shape (seq_len, batch_size).
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
            output = self.forward(src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)

            # Reshape for loss calculation
            output_reshaped = output.reshape(-1, output.size(-1))
            tgt_reshaped = tgt.reshape(-1)

            total_loss = self.criterion(output_reshaped, tgt_reshaped)

            # Calculate accuracy
            accuracy, predictions, ground_truths = calculate_accuracy(output_reshaped, tgt_reshaped)

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

