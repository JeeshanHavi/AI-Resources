import numpy as np
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from typing import List, Union

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def tokenize(sentence: str) -> List[str]:
    # Tokenizes a sentence into words.
    return nltk.word_tokenize(sentence)

def clean_text(text: str) -> str:
    # Cleans the input text by removing punctuation and special characters.
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def stem(word: str, method: str = 'porter') -> str:
    # Stems or lemmatizes a word based on the chosen method.
    word = word.lower()
    if method == 'porter':
        return stemmer.stem(word)
    elif method == 'lemmatization':
        return lemmatizer.lemmatize(word)
    else:
        raise ValueError("Method must be 'porter' or 'lemmatization'.")

def bag_of_words(tokenize_sentence: List[str], words: List[str], method: str = 'porter', ignore_case: bool = True) -> np.ndarray:
    # Creates a bag of words representation of the tokenized sentence
    if ignore_case:
        sentence_word = [stem(word, method) for word in tokenize_sentence]
    else:
        sentence_word = [stem(word, method) for word in tokenize_sentence]

    bag = np.zeros(len(words), dtype=np.float32)

    for idx, w in enumerate(words):
        if stem(w, method) in sentence_word:
            bag[idx] = 1

    return bag


    