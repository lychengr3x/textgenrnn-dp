"""
example for training
"""

from textgenrnn import textgenrnn
from datetime import datetime
import os


def run(model_name, dp, eta=None, clipnorm=None, dropout=0.0):
    assert isinstance(model_name, str)
    assert isinstance(dp, bool)
    assert isinstance(dropout, (int, float))
    if dp:
        assert eta is not None
        assert isinstance(eta, (int, float))
        assert clipnorm is not None
        assert isinstance(clipnorm, (int, float))

    print("Start training...\n")
    model_cfg = {
        'word_level': True,   # set to True if want to train a word-level model (requires more data and smaller max_length)
        'rnn_size': 128,   # number of LSTM cells of each layer (128/256 recommended)
        'rnn_layers': 3,   # number of LSTM layers (>=2 recommended)
        'rnn_bidirectional': False,   # consider text both forwards and backward, can give a training boost
        'max_length': 10,   # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
        'max_words': 10000,   # maximum number of words to model; the rest will be ignored (word-level model only)
    }

    if dp:
        train_cfg = {
            'line_delimited': True,   # set to True if each text has its own line in the source file
            'num_epochs': 10,   # set higher to train the model for longer
            'gen_epochs': 1,   # generates sample text from model after given number of epochs
            'train_size': 0.8,   # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
            'dropout': dropout,   # ignore a random proportion of source tokens each epoch, allowing model to generalize better
            'validation': True,   # If train__size < 1.0, test on holdout dataset; will make overall training slower
            'is_csv': False,   # set to True if file is a CSV exported from Excel/BigQuery/pandas
            'multi_gpu': False,
            'dp': dp, 
            'noise_eta': eta, 
            'noise_gamma': 0.55,
            'clipnorm': clipnorm,
        }
    else:
        train_cfg = {
            'line_delimited': True,   # set to True if each text has its own line in the source file
            'num_epochs': 10,   # set higher to train the model for longer
            'gen_epochs': 1,   # generates sample text from model after given number of epochs
            'train_size': 0.8,   # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
            'dropout': dropout,   # ignore a random proportion of source tokens each epoch, allowing model to generalize better
            'validation': True,   # If train__size < 1.0, test on holdout dataset; will make overall training slower
            'is_csv': False,   # set to True if file is a CSV exported from Excel/BigQuery/pandas
            'multi_gpu': False,
            'dp': dp, 
        }

    file_name = "data/ptb.train.txt"
    # model_name = "0221_1730_word"

    textgen = textgenrnn(name=model_name)

    train_function = (
        textgen.train_from_file
        if train_cfg["line_delimited"]
        else textgen.train_from_largetext_file
    )
    if dp:
        train_function(
            file_path=file_name,
            new_model=True,
            num_epochs=train_cfg["num_epochs"],
            gen_epochs=train_cfg["gen_epochs"],
            batch_size=1024,
            train_size=train_cfg["train_size"],
            dropout=train_cfg["dropout"],
            validation=train_cfg["validation"],
            is_csv=train_cfg["is_csv"],
            rnn_layers=model_cfg["rnn_layers"],
            rnn_size=model_cfg["rnn_size"],
            rnn_bidirectional=model_cfg["rnn_bidirectional"],
            max_length=model_cfg["max_length"],
            dim_embeddings=100,
            word_level=model_cfg["word_level"],
            dp=train_cfg["dp"],
            noise_eta=train_cfg['noise_eta'],
            noise_gamma=train_cfg['noise_gamma'],
            clipnorm=train_cfg['clipnorm'],
        )
    else:
        train_function(
            file_path=file_name,
            new_model=True,
            num_epochs=train_cfg["num_epochs"],
            gen_epochs=train_cfg["gen_epochs"],
            batch_size=1024,
            train_size=train_cfg["train_size"],
            dropout=train_cfg["dropout"],
            validation=train_cfg["validation"],
            is_csv=train_cfg["is_csv"],
            rnn_layers=model_cfg["rnn_layers"],
            rnn_size=model_cfg["rnn_size"],
            rnn_bidirectional=model_cfg["rnn_bidirectional"],
            max_length=model_cfg["max_length"],
            dim_embeddings=100,
            word_level=model_cfg["word_level"],
            dp=train_cfg["dp"],
        )
    print(f"Training finished for {model_name}\n\n\n")


if __name__ == "__main__":
    # no noise added
    # run(model_name="models/no_noise", dp=False)

    clipnorm = [0.7, 1, 0.7]
    dropout = [0, 0.2, 0.2]

    for clip, drop in zip(clipnorm, dropout):
        eta = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        model_name = [f"models/eta_{e}_gamma_0.55_clip_{clip}_dropout_{drop}" for e in eta]
        for e, m in zip(eta, model_name):
            run(model_name=m, dp=True, eta=e, clipnorm=clip, dropout=drop)