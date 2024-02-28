import random
import scipy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = 1024
SOS_TOKEN = 1025 
EOS_TOKEN = 1026 

def parse_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t', 1)
            utterance_id = parts[0]
            tokens = eval(parts[1])[0]
            data.append((utterance_id, tokens))
    return data

def read_word_seg_features(pth): 
    data = scipy.io.loadmat(pth)
    offset = 3

    utt2boundaries = {}
    cnt = 0 
    for key, mat in data.items(): 
        if cnt < offset: 
            cnt += 1
            continue 
        utt2boundaries[key] = mat[0][0][1][0]
    return utt2boundaries

class SpeechTokensDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing speech token data with word boundary information.
    
    The dataset supports extracting segments based on word boundaries and includes the functionality
    to provide context segments around a central segment. Each segment is de-duplicated to remove
    consecutive duplicates. Optionally, the dataset can work with ground truth or predicted word
    segmentation boundaries.

    Attributes:
        file_path (str): Path to the file containing speech tokens.
        word_seg_file_path (str): Path to the file containing word segmentation information.
        ground_truth_word_seg (bool): Flag indicating whether to use ground truth word segmentation. 
                                      If False, predicted word segmentation is used, which assumes 
                                      the start and end of the utterance as additional boundaries.
        segment_context_size (int): The number of segments to include as context for each central segment.
        mode (str): Operation mode of the dataset. Can be 'train' or 'eval' to adjust processing accordingly.

    The dataset returns a tuple (X, Y) for each item, where:
        X (torch.Tensor): The central segment of speech tokens, de-duplicated.
        Y (torch.Tensor): The context segments surrounding the central segment, each de-duplicated and
                          padded with PAD_TOKEN to match the maximum segment length within the context window.

    Example usage:
        dataset = SpeechTokensDataset(file_path='path/to/tokens.txt',
                                      word_seg_file_path='path/to/word_segments.txt',
                                      ground_truth_word_seg=True,
                                      segment_context_size=3,
                                      mode='train')
        for X, Y in dataset:
            # Process X and Y
    """
    def __init__(self, file_path, word_seg_file_path, ground_truth_word_seg=True, segment_context_size=3, mode='train', max_seq_len=512):
        self.data = parse_data(file_path)
        self.mode = mode
        self.utt2boundaries = read_word_seg_features(word_seg_file_path)
        self.ground_truth_word_seg = ground_truth_word_seg
        self.K = segment_context_size
        self.max_seq_len = max_seq_len - 2 # account for SOS and BOS token

    def __len__(self):
        return len(self.data)

    def crop_segment_length(self, segment_tensor): 
        # Crop sequences if longer than max_seq_len
        if segment_tensor.size(0) > self.max_seq_len:
            start_idx = random.randint(0, segment_tensor.size(0) - self.max_seq_len)
            return segment_tensor[start_idx:start_idx + self.max_seq_len]
        else: return segment_tensor

    def __getitem__(self, idx):
        uttid, tokens = self.data[idx]
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        predicted_boundaries = self.utt2boundaries[uttid]
        if self.ground_truth_word_seg: # for gt word boundaries, assume start and end as is
            boundaries = predicted_boundaries
        else: # for predicted word boundaries, always start at 0 and end at len(tokens)
            boundaries = [0] + predicted_boundaries + [len(tokens)]
        assert len(tokens) >= boundaries[-1], "Token length is shorter than the last boundary index"

        # Ensure there are enough boundaries to select a valid segment
        if len(boundaries) > 2*self.K + 1:
            # Randomly select a center segment index, ensuring it has K segments on either side
            segment_idx = random.randint(self.K, len(boundaries) - self.K - 2)
        else:
            # Default to the first segment if not enough segments for context
            segment_idx = 0

        # Initialize Y with padding values
        max_y_len = min(max([b - a for a, b in zip(boundaries[:-1], boundaries[1:])]), self.max_seq_len) + 1  # +1 for EOS
        Y = torch.full((2*self.K, max_y_len), PAD_TOKEN, dtype=torch.long)

        # Extract and de-duplicate segments for X and Y, handling edge cases
        for i in range(-self.K, self.K + 1):
            context_idx = segment_idx + i
            if i == 0:  # Center segment for X
                start_idx, end_idx = boundaries[segment_idx], boundaries[segment_idx + 1]
                X = self.deduplicate_segment(tokens_tensor[start_idx:end_idx])
                X = self.crop_segment_length(X)
                # Add SOS and EOS tokens for X
                X = torch.cat([torch.tensor([SOS_TOKEN]), X, torch.tensor([EOS_TOKEN])])
            else:  # Context segments for Y, with repetition for edge cases
                if context_idx < 0:
                    context_idx = 0  # Repeat the first segment if out of range on the left
                elif context_idx >= len(boundaries) - 1:
                    context_idx = len(boundaries) - 2  # Repeat the last segment if out of range on the right
                start_idx, end_idx = boundaries[context_idx], boundaries[context_idx + 1]
                context_segment = self.deduplicate_segment(tokens_tensor[start_idx:end_idx])
                context_segment = self.crop_segment_length(context_segment)
                # Add EOS tokens for each context_segment
                context_segment = torch.cat([context_segment, torch.tensor([EOS_TOKEN])])
                # Pad and place in Y
                if i < 0:
                    Y[self.K+i, :len(context_segment)] = context_segment
                else:
                    Y[self.K+i-1, :len(context_segment)] = context_segment

        return X, Y

    def deduplicate_segment(self, segment):
        """De-duplicate consecutive elements in a segment."""
        if segment.size(0) > 1:
            unique_mask = torch.cat([segment[:-1] != segment[1:], torch.tensor([True], dtype=torch.bool)])
            return segment[unique_mask]
        return segment


def collate_fn(batch):
    """
    Custom collate function for batching instances in the SpeechTokensDataset. This function
    prepares padded batches of source (X) and target (Y) sequences and generates several masks
    required for processing by the SpeechTransformerModel.

    Parameters:
    - batch (list of tuples): A list where each tuple corresponds to a data point from the 
      SpeechTokensDataset. Each tuple contains:
        - X (torch.Tensor): The source sequence tensor of size (T,), where T is the sequence length.
        - Y (torch.Tensor): The target sequence tensor of size (C, T), where C is the context size
          and T is the sequence length for each context.

    Returns:
    - Xs_padded (torch.Tensor): A batch of padded source sequences of size (B, T_max), where B is 
      the batch size and T_max is the length of the longest sequence in the batch.
    - Ys_flattened (torch.Tensor): A batch of flattened and padded target sequences of size 
      (B*C, T_max), where C is the context size for each sequence, effectively treating each context 
      segment as an independent sequence for decoding.
    - src_mask (NoneType): Placeholder for source sequence mask, typically not used in encoder self-attention 
      for tasks where each part of the source sequence should be visible to every other part. Set to `None`.
    - tgt_mask (torch.Tensor): A square binary mask of size (T_max, T_max) used to prevent attention to 
      future positions within each target context segment, enabling autoregressive decoding. The mask is 
      upper triangular with `True` values above the diagonal.
    - src_key_padding_mask (torch.Tensor): A boolean mask of size (B, T_max) indicating which elements 
      of the source sequences are padding and should be ignored by the attention mechanism.
    - tgt_key_padding_mask_flattened (torch.Tensor): A boolean mask of size (B*C, T_max) for the flattened target 
      sequences, indicating padding positions to be ignored during decoding.
    - memory_key_padding_mask (torch.Tensor): A boolean mask, cloned from `src_key_padding_mask`, 
      indicating positions in the encoder output to be ignored in the decoder's attention mechanism over 
      the encoder outputs.

    This function ensures that the model can efficiently handle variable-length sequences within a batch 
    and apply attention mechanisms correctly, respecting the autoregressive property in target sequences.
    """
    # Separate X and Y from the batch
    Xs = [item[0] for item in batch]  # List of all X tensors in the batch
    Ys = [item[1] for item in batch]  # List of all Y tensors in the batch

    # Pad X tensors. Since X is a 1D tensor, we use pad_sequence
    Xs_padded = pad_sequence(Xs, batch_first=True, padding_value=PAD_TOKEN)
  
	# Assuming no specific positions need to be masked within the source sequence, 
    # and the entire sequence can attend to itself fully:
    src_mask = None
 
    # Create src_key_padding_mask for X
    src_key_padding_mask = (Xs_padded == PAD_TOKEN)

    # Assuming Ys is a list of 2D tensors where each tensor has the same first dimension (context size)
    # Pad Y tensors to have uniform second dimension (segment length)
    max_seq_length = max(Y.size(1) for Y in Ys)  # Max length of any segment in context
    Ys_padded = torch.stack([torch.nn.functional.pad(Y, (0, max_seq_length - Y.size(1)), "constant", PAD_TOKEN) for Y in Ys])

    # Create tgt_key_padding_mask for Y
    tgt_key_padding_mask = (Ys_padded == PAD_TOKEN)  # Shape: (B, C, T)

    # Flatten Ys_padded for autoregressive processing in the model
    Ys_flattened = Ys_padded.view(-1, Ys_padded.size(-1))  # Shape: (B*C, T)

    # Adjust tgt_key_padding_mask for flattened Y
    tgt_key_padding_mask_flattened = tgt_key_padding_mask.view(-1, tgt_key_padding_mask.size(-1))  # Shape: (B*C, T)

    # Create tgt_mask for autoregressive decoding in flattened Y
    tgt_seq_len = Ys_flattened.size(1)
    tgt_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), dtype=torch.bool), diagonal=1)

    # memory_key_padding_mask can be the same as src_key_padding_mask if encoder outputs are used as is in the decoder
    memory_key_padding_mask = src_key_padding_mask.clone()
    
    return Xs_padded, Ys_flattened, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask_flattened, memory_key_padding_mask

def get_train_loader(file_path, word_seg_file_path, segment_context_size=3, batch_size=128, shuffle=True, num_workers=2, max_seq_len=512):
    dataset = SpeechTokensDataset(
        file_path=file_path, 
        word_seg_file_path=word_seg_file_path, 
        ground_truth_word_seg=True, 
        segment_context_size=segment_context_size, 
        mode='train', 
        max_seq_len=max_seq_len, 
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

def get_eval_loader(file_path, word_seg_file_path, segment_context_size=3, batch_size=128, shuffle=False, num_workers=2, max_seq_len=512):
    dataset = SpeechTokensDataset(
        file_path=file_path, 
        word_seg_file_path=word_seg_file_path, 
        ground_truth_word_seg=True, 
        segment_context_size=segment_context_size, 
        mode='eval', 
        max_seq_len=max_seq_len, 
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

