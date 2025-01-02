#include <iostream>
#include <vector>
#include <string>
#include <regex>
#include <sstream>
#include <algorithm>

// Simple Porter Stemmer implementation (you can replace with a full implementation)
std::string porter_stem(const std::string& word) {
    // A very basic stemming implementation for demonstration
    std::string stemmed = word;
    if (word.length() > 3 && word.substr(word.length() - 3) == "ing") {
        stemmed = word.substr(0, word.length() - 3);
    } else if (word.length() > 2 && word.substr(word.length() - 2) == "ed") {
        stemmed = word.substr(0, word.length() - 2);
    }
    return stemmed;
}

// Tokenize a sentence into words
std::vector<std::string> tokenize(const std::string& sentence) {
    std::vector<std::string> tokens;
    std::regex word_regex("[a-zA-Z0-9]+");
    auto words_begin = std::sregex_iterator(sentence.begin(), sentence.end(), word_regex);
    auto words_end = std::sregex_iterator();
    
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        tokens.push_back(i->str());
    }
    return tokens;
}

// Clean the input text by removing punctuation
std::string clean_text(const std::string& text) {
    return std::regex_replace(text, std::regex("[^a-zA-Z0-9\\s]"), "");
}

// Convert a sentence into a bag of words representation
std::vector<int> bag_of_words(const std::vector<std::string>& tokenize_sentence, const std::vector<std::string>& words, bool ignore_case = true) {
    std::vector<int> bag(words.size(), 0);
    
    // Create a set of stemmed words from the sentence
    std::vector<std::string> stemmed_sentence;
    for (const std::string& token : tokenize_sentence) {
        std::string word = token;
        if (ignore_case) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        }
        stemmed_sentence.push_back(porter_stem(word));
    }

    // Create a set for faster lookup
    std::unordered_set<std::string> sentence_set(stemmed_sentence.begin(), stemmed_sentence.end());

    // Mark the words present in the bag of words vector
    for (size_t idx = 0; idx < words.size(); ++idx) {
        std::string word = words[idx];
        if (ignore_case) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        }
        if (sentence_set.find(porter_stem(word)) != sentence_set.end()) {
            bag[idx] = 1;
        }
    }

    return bag;
}

int main() {
    // Example sentence and word list
    std::string sentence = "I am learning Natural Language Processing!";
    std::vector<std::string> words = {"learn", "natural", "language", "process", "hello", "world"};

    // Step 1: Tokenize the sentence
    std::vector<std::string> tokens = tokenize(sentence);

    // Step 2: Clean the sentence
    std::string cleaned_sentence = clean_text(sentence);
    std::cout << "Cleaned Sentence: " << cleaned_sentence << std::endl;

    // Step 3: Get Bag of Words
    std::vector<int> bag = bag_of_words(tokens, words, true);

    // Print the Bag of Words
    std::cout << "Bag of Words: ";
    for (int val : bag) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
