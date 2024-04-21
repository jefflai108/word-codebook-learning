import logging, os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from src.modules import (
    LabelSmoothingCrossEntropyLoss, 
    LearnedWeightedPooling, 
    AdaptiveConv1DPooling, 
    LearnableDictionaryEncoding, 
) 
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
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, max_seq_length, activation):
        super(TransformerEncoderModule, self).__init__()
        self.d_model = d_model
        # Initialize the linear layer for processing continuous embeddings
        self.input_linear = nn.Linear(input_dim, d_model, bias=False)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    def forward(self, src, src_mask, src_key_padding_mask):
        # Apply the linear layer to the continuous embeddings
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output

class TransformerDecoderModule(nn.Module):
    def __init__(self, embedding, d_model, nhead, num_decoder_layers, dim_feedforward, dropout, max_seq_length, vocab_size, activation):
        super(TransformerDecoderModule, self).__init__()
        self.embedding = embedding
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_seq_length)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
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
    def __init__(self, vocab_size, input_dim, num_layer_repre, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, max_seq_length, lr, activation='relu', word_pooling='mean', norm_type='batchnorm', label_smoothing=0.0, optimizer_type="sgd", logger=None):
        super(SpeechTransformerModel, self).__init__()

        self.logger = logger or logging.getLogger('SpeechTransformerModel')
        self.shared_embedding = nn.Embedding(vocab_size, d_model)
        self.num_layer_repre = num_layer_repre # M layer input repre, each has `input_dim` dim
        if num_layer_repre > 1: 
            self.layer_weights = nn.Parameter(torch.randn(self.num_layer_repre))
        self.encoder = TransformerEncoderModule(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, max_seq_length, activation)
        self.decoder = TransformerDecoderModule(self.shared_embedding, d_model, nhead, num_decoder_layers, dim_feedforward, dropout, max_seq_length, vocab_size, activation)

        # init word-pooling 
        self.word_pooling = word_pooling
        if self.word_pooling == 'weighted_mean': 
            self.word_pooling_layer = LearnedWeightedPooling(d_model) 
        elif self.word_pooling == 'conv_mean': 
            self.word_pooling_layer = AdaptiveConv1DPooling(d_model) 
        elif self.word_pooling == 'lde8': 
            self.word_pooling_layer = LearnableDictionaryEncoding(d_model, 8)
        elif self.word_pooling == 'lde16': 
            self.word_pooling_layer = LearnableDictionaryEncoding(d_model, 16)
        elif self.word_pooling == 'lde32': 
            self.word_pooling_layer = LearnableDictionaryEncoding(d_model, 32)

        # init word normalization scheme 
        self.norm_type = norm_type
        if norm_type == 'batchnorm':
            self.norm = nn.BatchNorm1d(d_model)
        elif norm_type == 'instancenorm':
            self.norm = nn.InstanceNorm1d(d_model)
        elif norm_type == 'l2norm':
            self.norm = None  # L2 normalization will be applied manually in forward
        elif norm_type == 'none':
            self.norm = None  # No normalization

        self.init_weights()

        # Optimizer selection
        if optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0)
        elif optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
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

        if self.num_layer_repre > 1: 
            # init weighted sum layer to all 1s 
            torch.nn.init.constant_(self.layer_weights, 1.0)  

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
   
    def repre_weighted_sum(self, input_repre): 
        """
        (normalized) weighted sum across input repres
        """
        seq_len, batch_size, _ = input_repre.size()
        # Reshape and permute input_repre to prepare for weighted sum
        input_repre = input_repre.view(seq_len, batch_size, self.num_layer_repre, -1)
        input_repre = input_repre.permute(0, 1, 3, 2)  # Bringing num_layer_repre to the last for easy multiplication

        # Normalize weights to sum to 1 using softmax
        norm_weights = F.softmax(self.layer_weights, dim=-1)

        # Apply weights and sum across the repre dimension
        # Note: Ensure norm_weights is correctly broadcastable to match input_repre's dimensions for multiplication
        # Output shape after weights_layer [seq_len, batch_size, input_dim]
        weighted_input_repre = (norm_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-2) * input_repre).sum(dim=-1)

        return weighted_input_repre

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
        if self.num_layer_repre > 1:
            # apply weighted sum (as in SUPERB's series) across different layer repres 
            src = self.repre_weighted_sum(src) 

        encoder_output = self.encoder(src, src_mask, src_key_padding_mask) # [seq_len, batch_size, hidden_dim]

        # obtain segment-level representation from encoder_output sequences. 
        # pooled_output is of shape [batch_size, hidden_dim]
        if self.word_pooling == 'mean': 
            pooled_output = self.encoder_masked_avg_pooling(encoder_output, src_key_padding_mask) 
        else: # 'weighted_mean' / 'conv_mean' / 'lde'
            pooled_output = self.word_pooling_layer(encoder_output, src_key_padding_mask)

        if self.norm_type == 'l2norm':
            pooled_output = F.normalize(pooled_output, p=2, dim=1)  # Apply L2 normalization
        elif self.norm is not None:
            pooled_output = self.norm(pooled_output)

        pooled_output = pooled_output.unsqueeze(0)
        output = self.decoder(tgt, pooled_output, tgt_mask, tgt_key_padding_mask, None) # need to adjust the "None" if pooled_output's seq_len is not 1 (to avoid attending to padded parts). But if seq_len is consistent / fixed across batches , then it's ok 

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
        if self.num_layer_repre > 1:
            # apply weighted sum (as in SUPERB's series) across different layer repres 
            src = self.repre_weighted_sum(src) 

        encoder_output = self.encoder(src, src_mask, src_key_padding_mask) # [seq_len, batch_size, hidden_dim]

        # obtain segment-level representation from encoder_output sequences. 
        # pooled_output is of shape [batch_size, hidden_dim]
        if self.word_pooling == 'mean': 
            pooled_output = self.encoder_masked_avg_pooling(encoder_output, src_key_padding_mask) 
        else: # 'weighted_mean' / 'conv_mean' / 'lde'
            pooled_output = self.word_pooling_layer(encoder_output, src_key_padding_mask)

        if self.norm_type == 'l2norm':
            pooled_output = F.normalize(pooled_output, p=2, dim=1)  # Apply L2 normalization
        elif self.norm is not None:
            pooled_output = self.norm(pooled_output)

        return pooled_output

    def train_step(self, src, tgt_input, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask, current_step, accumulation_steps=4):
        """
        Performs a single training step, incorporating gradient accumulation and accuracy calculation. 
        This method allows for simulating larger batch sizes by accumulating gradients over multiple steps, which is particularly 
        useful when hardware resources limit the maximum feasible batch size. After the specified number of accumulation 
        steps, it performs an optimization step and resets the gradients.

        Parameters:
        - src (Tensor): Source sequences tensor with shape (seq_len, batch_size).
        - tgt_input (Tensor): Input target sequences tensor with shape (seq_len, batch_size).
        - tgt (Tensor): Original target sequences tensor with shape (seq_len, batch_size).
        - src_mask (Tensor): Source sequence mask tensor.
        - tgt_mask (Tensor): Target sequence mask tensor.
        - src_key_padding_mask (Tensor): Source key padding mask tensor.
        - tgt_key_padding_mask (Tensor): Target key padding mask tensor.
        - memory_key_padding_mask (Tensor): Memory key padding mask tensor for attention mechanisms.
        - current_step (int): The current step number in the training loop.
        - accumulation_steps (int): The number of steps over which to accumulate gradients.

        Returns:
        - loss_value (float): The value of the loss for this step, scaled back to represent the actual loss over the accumulation period.
        - accuracy (float): The accuracy of the predictions for the current step.

        Note:
        - The optimization step and gradient zeroing occur only after the specified number of accumulation steps.
        """
        self.train()
        
        # Forward pass
        output = self.forward(src, tgt_input, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        
        # Reshape for loss calculation
        output_reshaped = output.reshape(-1, output.size(-1))
        tgt_reshaped = tgt.reshape(-1)
        
        # Compute loss and scale it for gradient accumulation
        loss = self.criterion(output_reshaped, tgt_reshaped) / accumulation_steps
        
        # Backpropagate the scaled loss
        loss.backward()
        
        # Accumulate gradients and perform optimization step at specified intervals
        if current_step % accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Calculate accuracy (this can be optionally accumulated or calculated at each step)
        with torch.no_grad():
            accuracy, predictions, ground_truths = calculate_accuracy(output_reshaped, tgt_reshaped, PAD_TOKEN)

        # Scale the loss back up to represent the total loss over the accumulated steps for logging or monitoring
        return loss.item() * accumulation_steps, (accuracy, predictions, ground_truths)

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
        Performs greedy decoding on the input source sequences using the model's encoder and decoder
        architectures. The method continues decoding until either the maximum specified length is reached
        or an EOS_TOKEN is generated, indicating the end of a sequence. Post EOS_TOKEN generation, further
        tokens are replaced with PAD_TOKEN to maintain tensor sizes without affecting sequence integrity.

        Parameters:
        - src (Tensor): The input tensor containing the source sequences. Shape is typically [seq_len, batch_size].
        - src_mask (Tensor): The mask for the source sequences, used in the attention mechanism of the encoder.
        - src_key_padding_mask (Tensor): The padding mask for the source sequences, indicating which elements
                                         are padding and should be ignored by the attention mechanisms.

        Returns:
        - Tensor: The tensor containing the generated sequences for each input in the batch. The sequences include
                  the initial SOS_TOKEN and are terminated by the first EOS_TOKEN. Any subsequent positions after
                  the EOS_TOKEN are filled with PAD_TOKENs to align with the longest sequence in the batch.

        Note:
        - The decoding process in each iteration predicts the next token based on the current state of the target
          tensor (tgt), which initially starts with SOS_TOKEN and is dynamically updated with each predicted token.
        - The function ensures that once a sequence generates an EOS_TOKEN, marking the logical end, it is
          deactivated, and subsequent tokens in the sequence are replaced with PAD_TOKENs, preventing any further
          meaningful output for that sequence. This ensures the final output correctly represents terminated sequences.
        """
        max_decoding_steps = max_length - 2

        self.eval()
        with torch.no_grad():
            # Prepare pooled_output_expanded as per the updated model's forward pass
            pooled_output = self.encoder_forward_pass(src, src_mask, src_key_padding_mask)
            pooled_output_expanded = pooled_output.unsqueeze(0)  # Ensure correct shape for decoder

            # Initialize the target tensor with SOS_TOKEN at the first position
            tgt = torch.full((1, src.shape[1]), SOS_TOKEN, dtype=torch.long, device=src.device)
            generated = tgt.clone()  # Ensure `generated` is a separate tensor

            # Initialize active mask to keep track of sequences still generating new tokens
            # It indicates if the sequence is still "activte" in generation. 
            # Once EOS_TOKEN is generated, the sequence deactivates permanently 
            active = torch.ones(src.shape[1], dtype=torch.bool, device=src.device)

            # Continue decoding until EOS_TOKEN is generated
            for _ in range(max_decoding_steps):

                # Call the decoder with the current state of tgt
                # No tgt_mask for greedy decoding as each step only looks at previously generated tokens
                output = self.decoder(tgt, pooled_output_expanded, None, None, None)
            
                # Select the last step's output for next token prediction
                next_token_logits = output[-1, :, :]  # Get logits of the last token
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # Predict the next token
                next_token = next_token.transpose(0, 1)

                # Append the predicted token to the generated sequence before any changes below 
                # this will ensure that the generated EOS_TOKEN is not replaced by PAD_TOKEN
                generated = torch.cat([generated, next_token], dim=0)

                # Check for EOS generation and update active status
                is_eos_or_pad = (next_token.squeeze(0) == EOS_TOKEN) | (next_token.squeeze(0) == PAD_TOKEN)
                active[is_eos_or_pad] = False  # Permanently deactivate sequences that generated EOS

                if not active.any():
                    break  # Stop decoding if all sequences are inactive

                # Replace tokens for inactive sequences with PAD_TOKEN to avoid influencing results
                next_token[~active, :] = PAD_TOKEN

                # Update tgt with the newly generated token for the next iteration
                tgt = torch.cat([tgt, next_token], dim=0)
        
        return generated

    def beam_search_decode(self, src, src_mask, src_key_padding_mask, max_length=50, beam_width=5):
        """
        Performs beam search decoding on the input source sequences. This method maintains multiple
        hypotheses (beams) and expands each by the possible next tokens at each step of decoding,
        only keeping the top scoring sequences based on cumulative log probabilities.

        Parameters:
        - src (Tensor): The input tensor containing the source sequences, shape [seq_len, batch_size].
        - src_mask (Tensor): The mask for the source sequences, used in the attention mechanism.
        - src_key_padding_mask (Tensor): The padding mask for the source sequences.
        - max_length (int): The maximum length of the sequence to decode, including the initial SOS_TOKEN.
        - beam_width (int): The number of hypotheses to maintain.

        Returns:
        - Tensor: The tensor containing the best generated sequence for each input in the batch.
                  Sequences are terminated by the first EOS_TOKEN. Any subsequent positions are
                  filled with PAD_TOKENs to maintain alignment.
        """
        max_decoding_steps = max_length - 2

        self.eval()
        with torch.no_grad():
            pooled_output = self.encoder_forward_pass(src, src_mask, src_key_padding_mask)
            pooled_output_expanded = pooled_output.unsqueeze(0)

            # Initialize beams with the SOS_TOKEN at the start and zero initial score
            init_token = torch.full((1, src.shape[1]), SOS_TOKEN, dtype=torch.long, device=src.device)
            beams = [(init_token, 0.0)]  # List of tuples (sequence tensor, cumulative log probability score)

            for step in range(max_decoding_steps):
                new_beams = []
                for seq, score in beams:
                    output = self.decoder(seq, pooled_output_expanded, None, None, None)
                    next_token_logits = output[-1, :, :]
                    log_probs, next_tokens = torch.topk(torch.log_softmax(next_token_logits, dim=-1), beam_width)

                    # Expand each current beam by the top-k next possible tokens
                    for i in range(beam_width):
                        next_token = next_tokens[:, i].unsqueeze(0).transpose(0, 1)
                        next_seq = torch.cat([seq, next_token], dim=0)
                        next_score = score + log_probs[:, i].item()
                        new_beams.append((next_seq, next_score))

                # Sort all expanded beams and keep only the top k overall
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            # Final selection of the best beam based on score and ensuring proper EOS termination
            completed_beams = [beam for beam in beams if beam[0][-1, 0] == EOS_TOKEN]
            if not completed_beams:
                completed_beams = beams  # Fall back to the best available beams if none properly terminate
            best_sequence = max(completed_beams, key=lambda x: x[1])[0]

            return best_sequence

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

