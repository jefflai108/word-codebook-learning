import unittest
import torch
from src.model import SpeechTransformerModel

PAD_TOKEN = 1024
SOS_TOKEN = 1025 
EOS_TOKEN = 1026 

class TestSpeechTransformerModel(unittest.TestCase):
    def setUp(self):
        # data hyper-parameters 
        self.batch_size = 8
        self.src_seq_len = 10 
        self.tgt_seq_len = 48 
        self.context_size = 3

        # Define model parameters
        self.vocab_size = 1024 + 3
        self.d_model = 512
        self.nhead = 8
        self.num_encoder_layers = 3
        self.num_decoder_layers = 3
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.max_seq_length = 120
        self.learning_rate = 0.01
        self.optimizer_type = "adam"

        # Initialize the model
        self.model = SpeechTransformerModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_seq_length=self.max_seq_length,
            lr=self.learning_rate,
            optimizer_type=self.optimizer_type
        )

    def test_forward_pass(self):
        # Create dummy data
        #Xs_padded : (4, 10) 
        #Ys_flattened : (24, 48) 
        #src_mask : None 
        #tgt_mask: (48, 48) 
        #src_key_padding_mask: (4, 10) 
        #tgt_key_padding_mask_flattened: (24, 48) 
        #memory_key_padding_mask: (4, 10) 
        
        src = torch.randint(0, self.vocab_size, (self.batch_size, self.src_seq_len))
        tgt = torch.randint(0, self.vocab_size, (self.batch_size * self.context_size, self.tgt_seq_len))
        src_mask = None 
        tgt_mask = torch.zeros((self.tgt_seq_len, self.tgt_seq_len)).type(torch.bool)
        src_key_padding_mask = torch.zeros(self.batch_size, self.src_seq_len).type(torch.bool)
        tgt_key_padding_mask = torch.zeros(self.batch_size * self.context_size, self.tgt_seq_len).type(torch.bool)
        memory_key_padding_mask = torch.zeros(self.batch_size, self.src_seq_len).type(torch.bool)

        # make the input [sequence_length, batch_size] instead 
        src = src.transpose(0, 1) 
        tgt = tgt.transpose(0, 1)

        # shift targets for teacher forcing 
        tgt_input = torch.cat([torch.full((1, tgt.shape[1]), SOS_TOKEN, dtype=tgt.dtype, device=tgt.device), tgt[:-1]], dim=0)

        # Forward pass
        output = self.model.forward(src, tgt_input, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)

        # Check output shape
        self.assertEqual(output.shape, (self.tgt_seq_len, self.batch_size * self.context_size, self.vocab_size))

    def test_training_step(self):
        src = torch.randint(0, self.vocab_size, (self.batch_size, self.src_seq_len))
        tgt = torch.randint(0, self.vocab_size, (self.batch_size * self.context_size, self.tgt_seq_len))
        src_mask = None 
        tgt_mask = torch.zeros((self.tgt_seq_len, self.tgt_seq_len)).type(torch.bool)
        src_key_padding_mask = torch.zeros(self.batch_size, self.src_seq_len).type(torch.bool)
        tgt_key_padding_mask = torch.zeros(self.batch_size * self.context_size, self.tgt_seq_len).type(torch.bool)
        memory_key_padding_mask = torch.zeros(self.batch_size, self.src_seq_len).type(torch.bool)

        # make the input [sequence_length, batch_size] instead 
        src = src.transpose(0, 1) 
        tgt = tgt.transpose(0, 1)

        # shift targets for teacher forcing 
        tgt_input = torch.cat([torch.full((1, tgt.shape[1]), SOS_TOKEN, dtype=tgt.dtype, device=tgt.device), tgt[:-1]], dim=0)

        initial_loss = self.model.train_step(src, tgt_input, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        for i in range(10):  # Train for a few steps
            loss, (acc, _, _) = self.model.train_step(src, tgt_input, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
            print(f"Iteration {i}: Loss = {loss:.3f}, Acc = {acc:.3f}")

    #def test_inference_step(self):
    #    src = torch.randint(0, self.vocab_size, (self.batch_size, self.src_seq_len))
    #    tgt = torch.randint(0, self.vocab_size, (self.batch_size * self.context_size, self.tgt_seq_len))
    #    src_mask = None 
    #    src_key_padding_mask = torch.zeros(self.batch_size, self.src_seq_len).type(torch.bool)

    #    # make the input [sequence_length, batch_size] instead 
    #    src = src.transpose(0, 1) 

    #    # Perform inference
    #    output = self.model.inference_step(src, src_mask, src_key_padding_mask)

    #    expected_output_shape = (self.batch_size, self.d_model)  # Adjust based on your model's expected output
    #    self.assertEqual(output.shape, expected_output_shape, "Output shape does not match expected shape.")
    #    self.assertFalse(torch.isnan(output).any(), "Output contains NaN values.")

    #def test_save_load_model(self):
    #    file_save_path = 'exp/debug/test_model.pth'
    #    self.model.save_model(1, file_save_path)

    #    new_model = SpeechTransformerModel(
    #        vocab_size=self.vocab_size,
    #        d_model=self.d_model,
    #        nhead=self.nhead,
    #        num_encoder_layers=self.num_encoder_layers,
    #        num_decoder_layers=self.num_decoder_layers,
    #        dim_feedforward=self.dim_feedforward,
    #        dropout=self.dropout,
    #        max_seq_length=self.max_seq_length,
    #        lr=self.learning_rate,
    #        optimizer_type=self.optimizer_type
    #    )
    #    epoch = new_model.load_model(file_save_path)

    #    self.assertEqual(epoch, 1, "Loaded model does not have the expected epoch.")

if __name__ == '__main__':
    unittest.main()

