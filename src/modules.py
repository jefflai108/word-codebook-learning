import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, layer_norm=True, apply_activation=True):
        super(ConvolutionalBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if layer_norm:
            self.layer_norm = nn.LayerNorm(out_channels)
        else:
            self.layer_norm = None
        self.apply_activation = apply_activation
        if apply_activation:
            self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        if self.layer_norm:
            x = x.transpose(1, 2)
            x = self.layer_norm(x)
            x = x.transpose(1, 2)
        if self.apply_activation:
            x = self.activation(x)
        return x

class AdaptiveConv1DPooling(nn.Module):
    """
    A custom pooling module that reduces the sequence length of encoder outputs by a factor of 4 using convolutional blocks,
    followed by adaptive average pooling to summarize the sequence into a single vector per batch item.

    The module uses two convolutional blocks, each with a stride of 2, to perform the downsampling. Each block consists of
    a convolutional layer followed by normalization and activation, which not only downsamples but also transforms the features.
    After reducing the dimensionality, an adaptive average pooling layer condenses the sequence to a fixed size output,
    ensuring that the final output is robust to varying input lengths and focused on the most salient features.

    Parameters:
        hidden_dim (int): The number of features in the input encoder outputs.
                          of the output after pooling.

    Forward Parameters:
        encoder_output (Tensor): The output from an encoder or previous network layer with shape [seq_len, batch_size, hidden_dim].
                                  This tensor contains the sequence data over which the pooling operation will be performed.
        src_key_padding_mask (Tensor): A boolean tensor of shape [batch_size, seq_len] where `True` indicates positions that
                                       are padded and should not be considered in the pooling operation. This mask ensures that
                                       padding does not affect the learned feature representations.

    Returns:
        Tensor: The output of the pooling operation with shape [batch_size, output_dim]. This tensor represents a condensed
                version of the input sequence where each batch item has been summarized into a single feature vector, making it
                suitable for further processing or prediction tasks.
    """
    def __init__(self, hidden_dim):
        super(AdaptiveConv1DPooling, self).__init__()
        self.layers = nn.ModuleList()
        input_dim = hidden_dim

        self.layers = nn.ModuleList([ # reduce sequence length by 4x
            ConvolutionalBlock(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, apply_activation=True),
            ConvolutionalBlock(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, apply_activation=False)  # No activation in the last block
        ])
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, encoder_output, src_key_padding_mask):
        # Permute encoder output to match [batch_size, hidden_dim, seq_len]
        encoder_output = encoder_output.permute(1, 2, 0)
        
        # Invert the mask: 'True' (PAD) becomes 0, 'False' (non-PAD) becomes 1
        mask_expanded = (~src_key_padding_mask).unsqueeze(1).expand(-1, encoder_output.size(1), -1).float()
        
        # Apply the mask
        encoder_output *= mask_expanded

        for layer in self.layers:
            encoder_output = layer(encoder_output)
        
        pooled_output = self.adaptive_pool(encoder_output).squeeze(-1)
        return pooled_output

class LearnedWeightedPooling(nn.Module):
    """
    Implements a learnable weighted pooling module where initial weights mimic mean pooling.

    This module uses a linear transformation to assign weights to each element in the input sequence
    based on their hidden state representation. Initially, weights are set to 1, resembling the 
    mean pooling operation where each input contributes equally. Weights are learnable and adjust during
    training to focus more on relevant elements of the input sequence.

    Parameters:
    - hidden_dim (int): The dimensionality of the input feature vector.

    Forward Parameters:
    - encoder_output (Tensor): The sequence of encoder outputs with shape [seq_len, batch_size, hidden_dim].
    - src_key_padding_mask (Tensor): A mask for the input sequence with shape [batch_size, seq_len],
      where `True` values indicate padding elements that should be ignored in pooling.

    Returns:
    - Tensor: The pooled vector of shape [batch_size, hidden_dim] after applying the learned weights
      to the encoder output.
    """
    def __init__(self, hidden_dim):
        super(LearnedWeightedPooling, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)
        # Initialize weights to all ones -- mimicing mean-pool at the beginning 
        nn.init.constant_(self.attention_weights.weight, 1.0)

    def forward(self, encoder_output, src_key_padding_mask):
        # encoder_output shape: [seq_len, batch_size, hidden_dim]
        # src_key_padding_mask shape: [batch_size, seq_len] with True for padding
        
        # Apply linear layer to each time step
        weights = self.attention_weights(encoder_output)  # [seq_len, batch_size, 1]
        weights = weights.squeeze(-1).permute(1, 0)  # [batch_size, seq_len]

        # Mask padding in the softmax computation to -inf before softmax
        weights = weights.masked_fill(src_key_padding_mask, float('-inf'))

        # Apply softmax to normalize weights to a probability distribution
        normalized_weights = F.softmax(weights, dim=1)  # [batch_size, seq_len]

        # Apply weights to the encoder outputs
        weighted_sum = torch.bmm(normalized_weights.unsqueeze(1), encoder_output.permute(1, 0, 2)).squeeze(1)
        # weighted_sum shape: [batch_size, hidden_dim]

        return weighted_sum

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.1, ignore_index=1024, reduction='mean'):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=-1)
        target = target.unsqueeze(-1)
        nll_loss = -log_probs.gather(dim=-1, index=target)
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = -log_probs.mean(dim=-1)
        
        # Apply label smoothing
        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            pad_mask = pad_mask.squeeze(-1)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
            count = (~pad_mask).sum()
        else:
            count = target.numel()

        loss = (1.0 - self.alpha) * nll_loss + self.alpha * smooth_loss

        if self.reduction == 'mean':
            loss = loss.sum() / count
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class LearnableDictionaryEncoding(nn.Module):
    """
    Implements the Learnable Dictionary Encoding (LDE) layer as described in the referenced paper,
    designed to encode a set of input vectors into a compact representation using a learnable dictionary
    and soft assignment weights. This encoding is useful for tasks such as feature extraction and
    dimensionality reduction in deep learning models.

    The LDE layer computes residuals between input vectors and dictionary atoms, determines occupancy
    probabilities (soft assignments) based on these residuals, and aggregates these weighted residuals
    to form a compact representation.

    The final output segment vector is obtained through a weighted attention mechanism across all the
    cluster mean vectors, where the weights are learned and applied to the mean vectors computed from
    the input embeddings and the dictionary atoms.

    Reference:
        "Learnable Dictionary Encoding" (https://arxiv.org/pdf/1804.00385.pdf)

    Parameters:
        feature_dim (int): The dimensionality of the input vectors and the dictionary atoms.
        num_of_clusters (int): The number of dictionary atoms, i.e., the size of the dictionary.

    Attributes:
        Dict (torch.nn.Parameter): The learnable dictionary initialized with random values.
                                   Shape: (num_of_clusters, feature_dim).
        cluster_proj (torch.nn.Linear or torch.nn.Identity): A projection layer to adjust the
                                                             dimensionality of input vectors to match
                                                             the dictionary atoms if necessary.
        soft_assignment_w (torch.nn.Parameter): Learnable weights for soft assignment of input vectors
                                                to dictionary atoms. Shape: (num_of_clusters,).
        soft_assignment_b (torch.nn.Parameter): Learnable biases for soft assignment.
                                                Shape: (num_of_clusters,).
        cluster_weights (torch.nn.Parameter): Learnable weights to compute the weighted attention
                                              across cluster mean vectors in the final output.

    Methods:
        forward(X): Computes the encoding of input tensor X through the LDE layer. Input X should
                    have the shape (B, L, D), where B is the batch size, L is the number of vectors
                    per batch, and D matches `feature_dim`.

                    Returns a tensor of encoded representations, where each sample in the batch is
                    represented as a weighted combination of cluster means. Output tensor shape is
                    determined by `num_of_clusters` and the encoding process.

    Example usage:
        >>> model = LearnableDictionaryEncoding(feature_dim=10, num_of_clusters=20)
        >>> X = torch.randn(5, 15, 10)  # Example input tensor
        >>> output = model(X)
        >>> print(output.shape)  # Expected shape depends on the implementation details
    """
    def __init__(self, feature_dim, num_of_clusters):
        super(LearnableDictionaryEncoding, self).__init__()
        self.num_of_clusters = num_of_clusters
            
        # Initialize the dictionary with random weights
        cluster_hidden_dim = feature_dim 
        self.Dict = nn.Parameter(torch.randn(num_of_clusters, cluster_hidden_dim))

        # cluster weighted mean 
        self.cluster_weights = nn.Parameter(torch.randn(self.num_of_clusters))

        self.init_weights()

        # projection layer 
        self.cluster_proj = nn.Identity()

        # Initialize learnable soft-assignment weights and biases
        self.soft_assignment_w = nn.Parameter(torch.ones(num_of_clusters))
        self.soft_assignment_b = nn.Parameter(torch.zeros(num_of_clusters))

    def init_weights(self):
        nn.init.uniform_(self.Dict, -1, 1)
  
        # init weighted clusters to all 1s
        nn.init.constant_(self.cluster_weights, 1.0)

    def cluster_weighted_sum(self, input_repre): 
        """
        (normalized) weighted sum across clusters
        """
        batch_size, num_of_clusters, cluster_hidden_dim = input_repre.size()
        # Reshape and permute input_repre to prepare for weighted sum
        input_repre = input_repre.permute(0, 2, 1)  

        # Normalize weights to sum to 1 using softmax
        norm_weights = F.softmax(self.cluster_weights, dim=-1)

        # Apply weights and sum across the repre dimension
        # Note: Ensure norm_weights is correctly broadcastable to match input_repre's dimensions for multiplication
        # Output shape after weights_layer [seq_len, batch_size, input_dim]
        weighted_input_repre = (norm_weights.unsqueeze(0).unsqueeze(0) * input_repre).sum(dim=-1)

        return weighted_input_repre

    def forward(self, X, src_key_padding_mask):
        # Permute X to match [batch_size, seq_len, feature_dim]
        X = X.permute(1, 0, 2)
        X = self.cluster_proj(X)
        B, L, D = X.shape

        # Step 1: compute Residual Tensor R
        R = X.unsqueeze(2) - self.Dict.unsqueeze(0).unsqueeze(0)  # shape (B, L, C, D)

        # Step 2: compute Occupancy Probability W
        # compute pairwise squared differences using broadcasting, then
        # sum over the feature dimension D to get squared L2 distances
        distance = ((R)**2).sum(-1) # shape (B, L, C)

        # Apply mask to distances by setting distances for padded elements to a large value
        distance_masked = distance.masked_fill(src_key_padding_mask.unsqueeze(-1), float(10000))

        # Eq (5) Compute soft assignments
        W = torch.softmax(-self.soft_assignment_w * distance_masked + self.soft_assignment_b, dim=-1) # shape (B, L, C)

        # Step 3: compute Aggregation Mat E
        E = (R * W.unsqueeze(-1)).sum(1) / L # Eq (7)

        # (normalized) weighted attention across all cluster mean vectors 
        S = self.cluster_weighted_sum(E) 

        return S
    
