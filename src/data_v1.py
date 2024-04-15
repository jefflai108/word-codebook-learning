import random
import scipy
import numpy as np 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = 1024
SOS_TOKEN = 1025 
EOS_TOKEN = 1026 

def parse_data(file_path, is_wav_file = False):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t', 1)
            utterance_id = parts[0]
            if is_wav_file: 
                values = parts[1]
            else: # values are list of tokens
                values = eval(parts[1])[0]
            data.append((utterance_id, values))
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


class Flicker8kSpeechDataset(Dataset):
    """
    A dataset class for handling the Flicker8k speech data, designed to support 
    training, evaluation, and inference modes for models that work with speech representations, 
    token sequences, and optional word segmentation boundaries. This class provides functionality 
    for processing and accessing speech data, including embeddings and tokens, with additional 
    support for handling ground truth or predicted word segmentation information.

    Parameters:
    - token_file_path (str): Path to the file containing tokens for each utterance.
    - embed_file_path (str): Path to the file containing embeddings for the utterances.
    - word_seg_file_path (str, optional): Path to the file containing word segmentation 
      features. If None, segmentation features are not used.
    - ground_truth_word_seg (bool, optional): Indicates whether to use ground truth word 
      segmentation. Defaults to True. If False, predicted segmentation is used.
    - segment_context_size (int, optional): The number of segments to consider on each side of 
      the target segment for context. Defaults to 3.
    - mode (str, optional): Operational mode of the dataset. Can be 'train', 'eval', or 
      'inference'. Defaults to 'train'.
    - max_seq_len (int, optional): The maximum length of a sequence after which it will be 
      cropped. Defaults to 512.

    The dataset provides support for dynamic segmentation of utterances based on the mode of 
    operation, handling special tokens, and ensuring sequences are within a specified length 
    limit through cropping and padding strategies.

	Usage:
    Initialize the dataset with paths to the necessary files and configuration parameters. 
    Access elements using indexing or iterate through the dataset to obtain processed utterance representations (X) 
    and context sequences (Y) tailored to the specified mode.

    In 'train' and 'eval' modes, this class facilitates the extraction of segments and their contextual neighbors, 
    providing a structured way to access speech representations and their corresponding tokens for model 
    training or evaluation. In 'inference' mode, entire utterances are processed, segmenting the speech 
    data for downstream prediction tasks.

    Example:
        >>> dataset = Flicker8kSpeechDataset(token_file_path='path/to/tokens.txt',
                                             embed_file_path='path/to/embeds.npy',
                                             word_seg_file_path='path/to/seg_features.txt',
                                             segment_context_size=2, mode='train')
        >>> print(len(dataset))
        >>> X, Y = dataset[0]

    Note:
    The `X` returned by the dataset is a 2D tensor representing the continuous speech embeddings of a segment, 
    while `Y` comprises sequences of discrete tokens (as 2D tensor) from neighboring segments, reflecting the 
    dataset's utility in modeling both continuous speech features and discrete linguistic tokens.
    """
    def __init__(self, token_file_path, embed_file_path, word_seg_file_path, ground_truth_word_seg=True, segment_context_size=3, mode='train', max_seq_len=512):
        self.tokens = parse_data(token_file_path)
        self.representations = np.load(embed_file_path, allow_pickle=True)
        self.mode = mode
        self.utt2boundaries = None if word_seg_file_path is None else read_word_seg_features(word_seg_file_path)
        self.ground_truth_word_seg = ground_truth_word_seg
        self.K = segment_context_size
        self.max_seq_len = max_seq_len - 2 # account for SOS and BOS token
        self.mode = mode 

    def __len__(self):
        return len(self.tokens)

    def crop_segment_length(self, segment_tensor): 
        # Crop sequences if longer than max_seq_len
        if segment_tensor.size(0) > self.max_seq_len:
            start_idx = random.randint(0, segment_tensor.size(0) - self.max_seq_len)
            return segment_tensor[start_idx:start_idx + self.max_seq_len]
        else: return segment_tensor

    def __getitem__(self, idx):
        uttid, tokens = self.tokens[idx]
        repre_tensor = torch.tensor(self.representations[uttid])
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)

        if self.utt2boundaries is None: 
            boundaries = [0, len(tokens)]
        else: 
            predicted_boundaries = self.utt2boundaries[uttid]
            if self.ground_truth_word_seg: # for gt word boundaries, assume start and end as is
                boundaries = predicted_boundaries
            else: # for predicted word boundaries, always start at 0 and end at len(tokens)
                boundaries = [0] + predicted_boundaries + [len(tokens)]
        assert len(tokens) >= boundaries[-1], "Token length is shorter than the last boundary index"

        # [inference mode] return entire utterances 
        if self.mode == 'inference':
            # Calculate max segment length for padding
            max_segment_len = max([b - a for a, b in zip(boundaries[:-1], boundaries[1:])]) 
            max_segment_len = min(max_segment_len, self.max_seq_len) 

            # Segment and pad
            segments = []
            for i in range(len(boundaries) - 1):
                start_idx, end_idx = boundaries[i], boundaries[i + 1]
                segment = repre_tensor[start_idx:end_idx]
                segment = self.crop_segment_length(segment) 
                padded_segment = torch.full((max_segment_len, repre_tensor.size(1)), 0.0, dtype=torch.float)
                padded_segment[:len(segment)] = segment
                segments.append(padded_segment)

            # Stack segments into a single tensor
            segments_tensor = torch.stack(segments)
            
            # number of segments, for decoding purpose 
            number_of_segments = len(boundaries) - 1
            assert segments_tensor.size(0) == number_of_segments
            
            return uttid, segments_tensor, number_of_segments

        # Determine segment index based on mode and context availability
        if self.mode == 'eval':
            # [eval, deterministic] always select the most central segment
            center_index = len(boundaries) // 2
            segment_idx = center_index
        else:
            # [train] handling based on the availability of enough context
            if len(boundaries) > 2*self.K + 1:
                # Randomly select a center segment index ensuring it has K segments on either side
                segment_idx = random.randint(self.K, len(boundaries) - self.K - 2)
            else:
                # Randomly select from available segments, ensuring some level of context is possible
                segment_idx = random.randint(0, max(0, len(boundaries) - 2))

        # Initialize Y with padding values
        max_y_len = min(max([b - a for a, b in zip(boundaries[:-1], boundaries[1:])]), self.max_seq_len) + 1  # +1 for EOS
        Y = torch.full((2*self.K, max_y_len), PAD_TOKEN, dtype=torch.long)

        # Extract and de-duplicate segments for X and Y, handling edge cases
        for i in range(-self.K, self.K + 1):
            context_idx = segment_idx + i
            if i == 0:  # Center segment for X
                start_idx, end_idx = boundaries[segment_idx], boundaries[segment_idx + 1]
                X = repre_tensor[start_idx:end_idx]
                X = self.crop_segment_length(X)
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
    A collate function for batching data with variable sequence lengths. It handles padding for batches of 
    sequences (Xs and Ys) to ensure they can be processed in parallel within models. The function pads X 
    sequences along the temporal dimension to match the longest sequence in the batch, and similarly pads Y 
    sequences for uniform length. Additionally, it prepares masks for attention mechanisms in transformer 
    models, including source masks, target masks, and key padding masks for both source and target sequences.

    Parameters:
    - batch (list of tuples): Each tuple contains X and Y tensors for a single example. X is assumed to be a 
      2D tensor with shape [sequence_length, feature_size], and Y is a list of 2D tensors where each tensor 
      represents context segments with variable lengths.

    Returns:
    - Xs_padded (Tensor): A 3D tensor of padded X sequences with shape [batch_size, max_sequence_length, feature_size].
    - Ys_flattened (Tensor): A 2D tensor of flattened and padded Y sequences for batch processing, with shape [(batch_size * context_size), max_segment_length].
    - src_mask (None): A placeholder for future source mask implementations, currently not applied.
    - tgt_mask (Tensor): A square, upper-triangular boolean tensor for masking future tokens in autoregressive decoding of Y, with shape [max_segment_length, max_segment_length].
    - src_key_padding_mask (Tensor): A boolean tensor indicating which elements of Xs_padded are padding, with shape [batch_size, max_sequence_length].
    - tgt_key_padding_mask_flattened (Tensor): A boolean tensor indicating padding positions in Ys_flattened, with shape [(batch_size * context_size), max_segment_length].
    - memory_key_padding_mask (Tensor): A clone of src_key_padding_mask for use in the decoder, indicating padding positions of encoder outputs.

    This function is designed to be used with PyTorch DataLoader for efficient batching of data with variable 
    sequence lengths, especially for models that incorporate attention mechanisms requiring padding masks.
    """
    # Separate X and Y from the batch
    Xs = [item[0] for item in batch]  # List of all X tensors in the batch
    Ys = [item[1] for item in batch]  # List of all Y tensors in the batch

    # Find the max length in the first dimension among all tensors
    max_len = max(x.size(0) for x in Xs)

    # Pad each tensor to have the same first dimension, max_len
    Xs_padded = [F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in Xs]

    # Stack to form a padded tensor 
    Xs_padded = torch.stack(Xs_padded, dim=0)
  
	# Assuming no specific positions need to be masked within the source sequence, 
    # and the entire sequence can attend to itself fully:
    src_mask = None
 
    # Create src_key_padding_mask for X --> FIXING 
    src_key_padding_mask = (Xs_padded == 0.0).any(dim=-1)

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

def inference_collate_fn(batch):
    """
    Prepares a batch of data for inference by padding segments to a uniform size and creating 
    a key padding mask to indicate the presence of padding within segments.

    Parameters:
    - batch (list of tuples): Each tuple contains an utterance ID, a tensor of speech segments,
      and the number of segments. The tensor has dimensions [num_segments, segment_length, feature_dim].

    Returns:
    - uttids (list): List of utterance IDs.
    - filtered_segments (Tensor): A 3D tensor of all speech segments after filtering out completely padded ones,
      reshaped to a uniform size for batch processing.
    - src_mask (None): Currently not used, placeholder for future sequence masking needs.
    - src_key_padding_mask (Tensor): A 2D boolean tensor indicating padding positions within each segment 
      for attention mechanisms.
    - segment_list (list): List containing the original number of segments for each utterance.
    """
    uttids = [item[0] for item in batch] 
    segment_list = [item[2] for item in batch] 

    # Find the maximum number of segments and max segment length across the batch
    max_num_segments = max([item[1].size(0) for item in batch])
    max_segment_len = max([item[1].size(1) for item in batch])
    feat_dim = [item[1].size(2) for item in batch][0]
    
    # Initialize a padded batch tensor with 0s
    padded_batch = torch.full((len(batch), max_num_segments, max_segment_len, feat_dim), 0.0, dtype=torch.float)
    
    # Loop through each item and copy its segments into the corresponding location in the padded batch tensor
    for i, (_, segments_tensor, _) in enumerate(batch):
        num_segments, segment_len, feat_dim = segments_tensor.size()
        padded_batch[i, :num_segments, :segment_len, :] = segments_tensor

    # Create a mask to identify non-padding segments
    not_padding_mask = ~(padded_batch == 0.0).all(dim=-1).all(dim=-1)  # Corrected to reduce over the last two dimensions
    not_padding_mask_flat = not_padding_mask.view(-1)  # Flatten for filtering

    # Reshape from (B, S, T, F) to (B*S, T, F) and filter
    reshaped_segments = padded_batch.view(-1, max_segment_len, feat_dim)
    filtered_segments = reshaped_segments[not_padding_mask_flat]

    src_mask = None
    src_key_padding_mask = (filtered_segments == 0.0).any(dim=-1)  # Flag individual padding tokens within segments
    
    return uttids, filtered_segments, src_mask, src_key_padding_mask, segment_list

def get_train_loader(token_file_path, embed_file_path, word_seg_file_path, segment_context_size=3, batch_size=128, shuffle=True, num_workers=2, max_seq_len=512):
    dataset = Flicker8kSpeechDataset(
        token_file_path=token_file_path, 
        embed_file_path=embed_file_path, 
        word_seg_file_path=word_seg_file_path, 
        ground_truth_word_seg=True, 
        segment_context_size=segment_context_size, 
        mode='train', 
        max_seq_len=max_seq_len, 
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

def get_eval_loader(token_file_path, embed_file_path, word_seg_file_path, segment_context_size=3, batch_size=128, shuffle=False, num_workers=2, max_seq_len=512):
    dataset = Flicker8kSpeechDataset(
        token_file_path=token_file_path, 
        embed_file_path=embed_file_path, 
        word_seg_file_path=word_seg_file_path, 
        ground_truth_word_seg=True, 
        segment_context_size=segment_context_size, 
        mode='eval', 
        max_seq_len=max_seq_len, 
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

def get_inference_loader(token_file_path, embed_file_path, word_seg_file_path, segment_context_size=3, batch_size=128, shuffle=False, num_workers=2, max_seq_len=512):
    dataset = Flicker8kSpeechDataset(
        token_file_path=token_file_path, 
        embed_file_path=embed_file_path, 
        word_seg_file_path=word_seg_file_path, 
        ground_truth_word_seg=True, 
        segment_context_size=segment_context_size, 
        mode='inference', 
        max_seq_len=max_seq_len, 
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=inference_collate_fn)

