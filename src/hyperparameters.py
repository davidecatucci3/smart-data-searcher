hyperparameters = {
    'batch_size': 4,
    'd_i': 768,             # dimension vector text encoder
    'd_t': 768,             # dimension vector image encoder
    'd_e': 512,              # dimension vector joint embedding
    'max_test_batches': 5,  # maximum number of batches to elaborate for testing
    'lr': 1e-4,
    'epochs': 8,
    'accumulation_steps': 8, # each how many steps weights are updated
    'max_length_txt': 128,   # max length raw text for text encoder
    'alpha': 0.8,
    'top_k': 5               # number of images selected from search (the top 5)
}