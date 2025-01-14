#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <regex>

class PorterStemmer {
public:
    std::string stem(const std::string& word) {
        // implement Porter stemmer algorithm
    }
};

class WordNetLemmatizer {
public:
    std::string lemmatize(const std::string& word) {
        // implement WordNet lemmatizer algorithm
    }
};

std::vector<std::string> tokenize(const std::string& sentence) {
    std::vector<std::string> words;
    std::string word;
    for (char c : sentence) {
        if (c == ' ') {
            if (!word.empty()) {
                words.push_back(word);
                word.clear();
            }
        } else {
            word += c;
        }
    }
    if (!word.empty()) {
        words.push_back(word);
    }
    return words;
}

std::string cleanText(const std::string& text) {
    std::regex pattern("[^a-zA-Z0-9\\s]");
    return std::regex_replace(text, pattern, "");
}

std::string stem(const std::string& word, const std::string& method) {
    PorterStemmer stemmer;
    WordNetLemmatizer lemmatizer;
    std::string lowerWord = word;
    std::transform(lowerWord.begin(), lowerWord.end(), lowerWord.begin(), ::tolower);
    if (method == "porter") {
        return stemmer.stem(lowerWord);
    } else if (method == "lemmatization") {
        return lemmatizer.lemmatize(lowerWord);
    } else {
        throw std::invalid_argument("Method must be 'porter' or 'lemmatization'.");
    }
}

std::vector<float> bagOfWords(const std::vector<std::string>& tokenizeSentence, const std::vector<std::string>& words, const std::string& method, bool ignoreCase) {
    std::vector<std::string> sentenceWord;
    for (const std::string& word : tokenizeSentence) {
        sentenceWord.push_back(stem(word, method));
    }
    std::vector<float> bag(words.size(), 0.0f);
    for (size_t idx = 0; idx < words.size(); ++idx) {
        if (std::find(sentenceWord.begin(), sentenceWord.end(), stem(words[idx], method)) != sentenceWord.end()) {
            bag[idx] = 1.0f;
        }
    }
    return bag;
}

int main() {
    PorterStemmer stemmer;
    WordNetLemmatizer lemmatizer;
    // use the functions
    return 0;
}
