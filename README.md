# Preprocessing the Cornell Movie Dialogs Corpus Dataset for Chatbot
This code snippet demonstrates how to preprocess the Cornell Movie Dialogs Corpus dataset for training a chatbot. The dataset contains conversational exchanges between movie characters and is used to create dialog pairs for the chatbot model.

# Preprocessing Steps
Loading the Dialog Data: The code reads the dialog data from the movie_lines.txt file in the specified directory.

# Creating Conversational Pairs: 
The dialog data is processed to create conversational pairs by grouping dialog lines based on conversation IDs.

# Tokenization: 
The code tokenizes the dialog pairs by converting the text to lowercase and splitting it into individual words using the NLTK library's word_tokenize function. An <EOS> (End-of-Sentence) token is added at the end of each input sentence.

# Filtering by Length: 
Dialog pairs exceeding a specified maximum length are filtered out to control the input sequence length for the chatbot.

# Building Vocabulary:
A vocabulary dictionary is constructed by counting the occurrences of words in the tokenized dialog pairs. Words with counts below a specified minimum threshold are excluded from the vocabulary.

# Converting Tokens to Indices:
 Each token in the tokenized dialog pairs is converted to its corresponding index in the vocabulary dictionary. The filtered tokenized pairs are converted to pairs of indexed tokens.

# Saving Preprocessed Data: 
The preprocessed data, including indexed dialog pairs, vocabulary, word-to-index mapping, and index-to-word mapping, is saved using pickle files.

# Usage
Set the data_path variable to the directory path where the movie_lines.txt file is located.

Set the save_path variable to the directory path where you want to save the preprocessed data.

Adjust the min_count and max_len parameters as needed to control vocabulary size and maximum input sequence length.

Run the code.

The preprocessed data, including indexed pairs and vocabulary files, will be saved in the specified save_path.

# Dependencies
Python 3.x
NLTK
Pickle
Credits
Cornell Movie Dialogs Corpus: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
NLTK: https://www.nltk.org/
Pickle: https://docs.python.org/3/library/pickle.html
