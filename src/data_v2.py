import random
import scipy
import numpy as np 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoFeatureExtractor, AutoModel

from src.util import trim_pad_tokens, filter_padded_rows

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
                                             segment_context_size=2)
        >>> print(len(dataset))
        >>> X, Y = dataset[0]

    Note:
    The `X` returned by the dataset is a 2D tensor representing the continuous speech embeddings of a segment, 
    while `Y` comprises sequences of discrete tokens (as 2D tensor) from neighboring segments, reflecting the 
    dataset's utility in modeling both continuous speech features and discrete linguistic tokens.
    """
    def __init__(self, token_file_path, embed_file_path, word_seg_file_path, ground_truth_word_seg=True, max_seq_len=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokens = parse_data(token_file_path)
        self.representations = np.load(embed_file_path, allow_pickle=True)
        #self.waveforms = np.load(waveform_file_path, allow_pickle=True)
        #self.repre_layer_indexes = int(repre_layer_indexes) # [0, 25] as 0 is CNN outputs
        #self.setup_representation_extractor(representation_path)
        self.utt2boundaries = None if word_seg_file_path is None else read_word_seg_features(word_seg_file_path)
        self.ground_truth_word_seg = ground_truth_word_seg
        self.max_seq_len = max_seq_len - 2 # account for SOS and BOS token

    #def setup_representation_extractor(self, representation_path): 
    #    self.repre_main_model = AutoModel.from_pretrained(representation_path)
    #    self.repre_feature_extractor = AutoFeatureExtractor.from_pretrained(representation_path)

    #    self.repre_main_model.config.output_hidden_states = True
    #    self.repre_main_model.eval() # disable layerdrop 
    #  
    #    self.repre_main_model = self.repre_main_model.to(self.device)

    #def extract_pretrained_repre(self, waveform):  
    #    with torch.no_grad():
    #        input_tensor = self.repre_feature_extractor(waveform, return_tensors="pt", sampling_rate = 16000).input_values.float()
    #        input_tensor = input_tensor.float().to(self.device)
    #        outputs = self.repre_main_model(input_tensor)
    #        layer_embeddings = outputs.hidden_states[self.repre_layer_indexes].squeeze()
    #    return layer_embeddings

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
        #waveform = self.waveforms[uttid]
        #repre_tensor = self.extract_pretrained_repre(waveform)
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
    
        ## [train, eval, inference] : return entire utterance 
        # Calculate max segment length for padding
        max_segment_len = max([b - a for a, b in zip(boundaries[:-1], boundaries[1:])]) 
        max_segment_len_x = min(max_segment_len, self.max_seq_len) 
        max_segment_len_y = max_segment_len_x + 1 # +1 for EOS

        # Segment and pad
        X_segments, Y_segments = [], []
        for i in range(len(boundaries) - 1):
            start_idx, end_idx = boundaries[i], boundaries[i + 1]

            X_segment = repre_tensor[start_idx:end_idx]
            X_segment = self.crop_segment_length(X_segment) 

            Y_segment = self.deduplicate_segment(tokens_tensor[start_idx:end_idx])
            Y_segment = self.crop_segment_length(Y_segment)
            # Add EOS tokens for each Y_segment
            Y_segment = torch.cat([Y_segment, torch.tensor([EOS_TOKEN])])

            X_padded_segment = torch.full((max_segment_len_x, repre_tensor.size(1)), 0.0, dtype=torch.float)
            Y_padded_segment = torch.full((max_segment_len_y, ), PAD_TOKEN, dtype=torch.long)

            X_padded_segment[:len(X_segment)] = X_segment
            Y_padded_segment[:len(Y_segment)] = Y_segment 

            X_segments.append(X_padded_segment)
            Y_segments.append(Y_padded_segment)

        # Stack segments into a single tensor
        X_segments_tensor = torch.stack(X_segments)
        Y_segments_tensor = torch.stack(Y_segments)
        Y_segments_tensor = trim_pad_tokens(Y_segments_tensor, PAD_TOKEN)

        # number of segments, for decoding purpose 
        number_of_segments = len(boundaries) - 1
        assert X_segments_tensor.size(0) == number_of_segments
        assert Y_segments_tensor.size(0) == number_of_segments

        return X_segments_tensor, Y_segments_tensor, uttid 

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
    # Unpack the batched items
    X_batch = [item[0] for item in batch]  # Collect all X_segments_tensor
    Y_batch = [item[1] for item in batch]  # Collect all Y_segments_tensor
    uttids = [item[2] for item in batch] 

    # Determine the maximum size in each dimension across all items for X and Y
    max_segments = max(x.size(0) for x in X_batch)
    max_seq_len_x = max(x.size(1) for x in X_batch)
    feature_size_x = X_batch[0].size(2)  
    max_seq_len_y = max(y.size(1) for y in Y_batch)

    # Initialize padded tensors with zeros for X and PAD_TOKEN for Y
    X_padded = torch.zeros((len(X_batch), max_segments, max_seq_len_x, feature_size_x))
    Y_padded = torch.full((len(Y_batch), max_segments, max_seq_len_y), PAD_TOKEN, dtype=torch.long)

    # Fill in the original data into the padded tensors
    for i, (x, y) in enumerate(zip(X_batch, Y_batch)):
        X_padded[i, :x.size(0), :x.size(1), :] = x
        Y_padded[i, :y.size(0), :y.size(1)] = y

    # X_padded shape : (batch_size, num_segment, segment_len, repre_dim)
    # Y_padded shape : (batch_size, num_segment, segment_len) 

    # Merge batch and num_segment dimensions, as we treat each segment as a training example 
    # and we do not utilize sequential ordering across segments in training 
    X_flattened = X_padded.view(-1, max_seq_len_x, feature_size_x)
    Y_flattened = Y_padded.view(-1, max_seq_len_y)

    # filter out rows in X and Y that are entirely padded. This can reduce GPU mem significantly 
    X_flattened, Y_flattened = filter_padded_rows(X_flattened, Y_flattened, padding_x=0.0, padding_y=PAD_TOKEN)

    # futher limit the number of segments to 1k per batch to avoid GPU mem error 
    # this likely won't be a concern for non-determistic inference results as we set the inference batch_size=1 
    max_samples = 1000 
    if X_flattened.size(0) > max_samples:
        # Randomly select max_samples indices
        indices = torch.randperm(X_flattened.size(0))[:max_samples]

        # Apply the same selection to both X and Y
        X_flattened = X_flattened[indices]
        Y_flattened = Y_flattened[indices]

    # Assuming no specific positions need to be masked within the source sequence, 
    # and the entire sequence can attend to itself fully:
    src_mask = None
 
    # Create src_key_padding_mask
    src_key_padding_mask = (X_flattened == 0.0).any(dim=-1)

    # For Y, the mask is where the values equal PAD_TOKEN
    tgt_key_padding_mask = (Y_flattened == PAD_TOKEN)

    # Create tgt_mask for autoregressive decoding in Y
    tgt_seq_len = Y_flattened.size(1)
    tgt_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), dtype=torch.bool), diagonal=1)

    # memory_key_padding_mask can be the same as src_key_padding_mask for simplicity
    memory_key_padding_mask = src_key_padding_mask.clone()
   
    #print(X_flattened.shape, Y_flattened.shape)
    return uttids, X_flattened, Y_flattened, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask

def get_train_loader(token_file_path, embed_file_path, word_seg_file_path, batch_size=128, shuffle=True, num_workers=2, max_seq_len=512):
    dataset = Flicker8kSpeechDataset(
        token_file_path=token_file_path, 
        embed_file_path=embed_file_path, 
        word_seg_file_path=word_seg_file_path, 
        ground_truth_word_seg=True, 
        max_seq_len=max_seq_len, 
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

def get_eval_loader(token_file_path, embed_file_path, word_seg_file_path, batch_size=128, shuffle=False, num_workers=2, max_seq_len=512):
    dataset = Flicker8kSpeechDataset(
        token_file_path=token_file_path, 
        embed_file_path=embed_file_path, 
        word_seg_file_path=word_seg_file_path, 
        ground_truth_word_seg=True, 
        max_seq_len=max_seq_len, 
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

def get_inference_loader(token_file_path, embed_file_path, word_seg_file_path, batch_size=128, shuffle=False, num_workers=2, max_seq_len=512):
    dataset = Flicker8kSpeechDataset(
        token_file_path=token_file_path, 
        embed_file_path=embed_file_path, 
        word_seg_file_path=word_seg_file_path, 
        ground_truth_word_seg=True, 
        max_seq_len=max_seq_len, 
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

