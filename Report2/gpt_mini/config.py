import torch

DEFAULT_DEVICE = "cpu"

if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"


CONFIG = {
    "preprocess": {
        "dataset": 'datasets/lmd_full',
        # The number of events we're going to extract
        # currently just from the start of the file
        "window_size": 256,  # (48)
        "new_dataset_index": "training_data.txt",
        "data_train": "data_train.txt",
        "data_test": "data_test.txt",
        # Stop after processing this many items from
        # the original dataset
        "max_dataset_items": 8000,
    },
    "tokenizer": {
        "vocab_size": 50257,
        "model": "./checkpoints/midi_tokenizer_bpe.pkl"
    },
    "model": {
        # "model_type": "gpt-nano",
        # See model.py for suggestions
        "n_layer": 6,
        "n_head": 6,
        "n_embed": 192,
        ##
        "vocab_size": 50257,
        "block_size": 256,
        # dropout hyperparameters
        "embd_pdrop": 0.1,
        "resid_pdrop": 0.1,
        "attn_pdrop": 0.1,
    },
    "training": {
        "workers": 6,
        "max_iters": 1000,
        "batch_size": 32,
        "learning_rate": 0.03,
        # device set above
    }
}

