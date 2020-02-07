"""
example for training
"""

from textgenrnn import textgenrnn
from datetime import datetime
import os


def main():
    model_cfg = {
        "word_level": False,  # set to True if want to train a word-level model (requires more data and smaller max_length)
        "rnn_size": 128,  # number of LSTM cells of each layer (128/256 recommended)
        "rnn_layers": 3,  # number of LSTM layers (>=2 recommended)
        "rnn_bidirectional": False,  # consider text both forwards and backward, can give a training boost
        "max_length": 30,  # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
        "max_words": 10000,  # maximum number of words to model; the rest will be ignored (word-level model only)
    }

    train_cfg = {
        "line_delimited": False,  # set to True if each text has its own line in the source file
        "num_epochs": 60,  # set higher to train the model for longer
        "gen_epochs": 5,  # generates sample text from model after given number of epochs
        "train_size": 0.8,  # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
        "dropout": 0.0,  # ignore a random proportion of source tokens each epoch, allowing model to generalize better
        "validation": False,  # If train__size < 1.0, test on holdout dataset; will make overall training slower
        "is_csv": False,  # set to True if file is a CSV exported from Excel/BigQuery/pandas
        "multi_gpu": False,
        "dp": True,
        "lr": 4e-3,
        "l2_norm_clip": 1.0,
        "noise_multiplier": 1.1,
        "num_microbatches": 128,
    }

    file_name = "data/ptb.train.txt"
    model_name = "0206_2018"

    textgen = textgenrnn(name=model_name)

    train_function = (
        textgen.train_from_file
        if train_cfg["line_delimited"]
        else textgen.train_from_largetext_file
    )

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
        lr=train_cfg["lr"],
        l2_norm_clip=train_cfg["l2_norm_clip"],
        noise_multiplier=train_cfg["noise_multiplier"],
        num_microbatches=train_cfg["num_microbatches"],
    )


if __name__ == "__main__":
    main()
