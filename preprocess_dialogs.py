import os
import re
import nltk
import pickle
import numpy as np

from nltk.tokenize import word_tokenize
from collections import Counter


def preprocess_dialogs(data_path, save_path, min_count=5, max_len=20):
    # Load the dialog data
    dialog_file = os.path.join(data_path, 'movie_lines.txt')
    with open(dialog_file, 'r', encoding='iso-8859-1') as f:
        dialog_lines = f.readlines()

    # Create a dictionary of conversations
    conversations = {}
    for line in dialog_lines:
        line_parts = line.split(' +++$+++ ')
        conv_id = line_parts[1]
        text = line_parts[4]
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(text.strip())

    # Create a list of dialog pairs
    dialog_pairs = []
    for conv_id in conversations:
        conv = conversations[conv_id]
        for i in range(len(conv)-1):
            dialog_pairs.append((conv[i], conv[i+1]))

    # Tokenize the dialog pairs
    tokenized_pairs = []
    for pair in dialog_pairs:
        tokens = word_tokenize(pair[0].lower()) + ['<EOS>']
        if len(tokens) <= max_len:
            tokenized_pairs.append((tokens, pair[1]))

    # Create a vocabulary dictionary
    word_counts = Counter([token for pair in tokenized_pairs for token in pair[0]])
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    vocab_size = len(vocab)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    # Convert tokens to indices
    indexed_pairs = []
    for tokens, response in tokenized_pairs:
        indexed_tokens = [word_to_idx[token] for token in tokens if token in vocab]
        indexed_pairs.append((indexed_tokens, response))

    # Save the preprocessed data
    with open(os.path.join(save_path, 'indexed_pairs.pkl'), 'wb') as f:
        pickle.dump(indexed_pairs, f)
    with open(os.path.join(save_path, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)
    with open(os.path.join(save_path, 'word_to_idx.pkl'), 'wb') as f:
        pickle.dump(word_to_idx, f)
    with open(os.path.join(save_path, 'idx_to_word.pkl'), 'wb') as f:
        pickle.dump(idx_to_word, f)

    return indexed_pairs, vocab, word_to_idx, idx_to_word
