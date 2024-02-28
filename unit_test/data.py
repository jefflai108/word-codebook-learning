import unittest
import torch
from torch.utils.data import DataLoader
from src.data import SpeechTokensDataset, collate_fn, inference_collate_fn

PAD_TOKEN = 1024
SOS_TOKEN = 1025 
EOS_TOKEN = 1026 

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.test_file_path = '/data/sls/scratch/clai24/word-seg/flicker8k/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_test.txt'
        self.word_seg_file_path = '/data/sls/scratch/sbhati/data/flicker/flicker_speech_features.mat'
        self.batch_size = 4 
        self.vocab_size = 1024 + 3 # account for special tokens 
        self.segment_context_size = 3
        self.dataset = SpeechTokensDataset(
            file_path=self.test_file_path, 
            word_seg_file_path=self.word_seg_file_path, 
            ground_truth_word_seg=True, 
            segment_context_size=self.segment_context_size, 
            mode='train'
        )
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=collate_fn)
        self.inference_dataset = SpeechTokensDataset(
            file_path=self.test_file_path, 
            word_seg_file_path=self.word_seg_file_path, 
            ground_truth_word_seg=True, 
            segment_context_size=self.segment_context_size, 
            mode='inference'
        )
        self.inference_dataloader = DataLoader(self.inference_dataset, batch_size=self.batch_size, collate_fn=inference_collate_fn)

    def test_inference_loader(self): 
        for uttids, X, _, _ in self.inference_dataloader:
            #import pdb; pdb.set_trace()
            self.assertEqual(len(X.shape), 2, "Returned tensor is not 2D")

    def test_vocab_distribution(self): 
        vocab_counts_accumulated = torch.zeros(self.vocab_size, dtype=torch.long)
        for X, Y, _, _, _, _, _ in self.dataloader:
            X_flat = X.flatten()
            batch_counts = torch.bincount(X_flat, minlength=self.vocab_size)
            vocab_counts_accumulated += batch_counts

        # Sort and print final accumulated distribution
        sorted_indices_accumulated = torch.argsort(vocab_counts_accumulated, descending=True)
        sorted_indices_non_zero_accumulated = sorted_indices_accumulated[vocab_counts_accumulated[sorted_indices_accumulated] > 0]
        sorted_counts_accumulated = vocab_counts_accumulated[sorted_indices_non_zero_accumulated]

        # Print final vocab distribution from most to least occurrences
        vocab_distribution = [(int(vocab_index), int(count)) for vocab_index, count in zip(sorted_indices_non_zero_accumulated, sorted_counts_accumulated)]
        print(vocab_distribution)

    def test_vocab_size(self): 
        max_value = 0 
        for X, Y, _, _, _, _, _ in self.dataloader:
            max_value = max(torch.max(X).item(), max_value)
            max_value = max(torch.max(Y).item(), max_value)
        print(f'max value is {max_value}') # max value should be the PAD_TOKEN 
        assert max_value == self.vocab_size - 1

    #def test_batch_structure(self):
    #    for X, Y, _, _, _, _, _ in self.dataloader:
    #        batch_size = X.size(0)
    #        # Check X and Y are tensors
    #        self.assertTrue(isinstance(X, torch.Tensor), "X is not a tensor")
    #        self.assertTrue(isinstance(Y, torch.Tensor), "Y is not a tensor")
    #        
    #        # Check Y dimensions
    #        self.assertEqual(Y.size(0), batch_size * 2 * self.segment_context_size, "Y's context size is incorrect")

    #        # Check for padding value in Y
    #        self.assertTrue((Y == PAD_TOKEN).any(), "Y does not contain padding values where expected")

    ## Example method to check de-duplication in X
    #def test_deduplication(self):
    #    for X, Y, _, _, _, _, _ in self.dataloader:
    #        # Ensure no consecutive elements in X are identical, ignoring padding values
    #        for batch in X:
    #            # Convert to a list for easier manipulation
    #            x_list = batch.tolist()
    #            # Filter out padding values before checking for consecutive duplicates
    #            filtered_x = [x for x in x_list if x != PAD_TOKEN]
    #            # Check that no consecutive elements are the same in the filtered list
    #            self.assertTrue(all(filtered_x[i] != filtered_x[i + 1] for i in range(len(filtered_x) - 1)), "X contains consecutive duplicates")

    #def test_masks_generation(self):
    #    for X, Y, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask in self.dataloader:
    #        # Test src_key_padding_mask correctness
    #        self.assertEqual(X.size(0), src_key_padding_mask.size(0), "src_key_padding_mask has incorrect batch size")
    #        self.assertTrue((src_key_padding_mask.sum(dim=1) == (X == PAD_TOKEN).sum(dim=1)).all(), "src_key_padding_mask incorrectly generated")

    #        # Test tgt_key_padding_mask correctness
    #        # Assuming Y is flattened for simplicity in explaining; adjust as per your actual implementation
    #        self.assertEqual(Y.size(0), tgt_key_padding_mask.size(0), "tgt_key_padding_mask has incorrect batch size")
    #        self.assertTrue((tgt_key_padding_mask.sum(dim=1) == (Y == PAD_TOKEN).sum(dim=1)).all(), "tgt_key_padding_mask incorrectly generated")

    #        # Test tgt_mask for preventing future attention
    #        # Check that each row in tgt_mask has the correct pattern of False followed by True values
    #        for i in range(1, tgt_mask.size(0)):
    #            # True values should start appearing at position (i, i+1) and continue to the end of the row
    #            true_start_index = i + 1
    #            if true_start_index < tgt_mask.size(1):  # Ensure index is within bounds
    #                # Check that all values before true_start_index are False and after are True
    #                self.assertTrue(tgt_mask[i, :true_start_index].all() == False, "Incorrect False masking before future positions")
    #                self.assertTrue(tgt_mask[i, true_start_index:].all() == True, "Incorrect True masking for future positions")
    #        
    #        # Assuming memory_key_padding_mask is intended to be identical to src_key_padding_mask
    #        self.assertTrue(torch.equal(src_key_padding_mask, memory_key_padding_mask), "memory_key_padding_mask does not match src_key_padding_mask")

if __name__ == '__main__':
    unittest.main()
