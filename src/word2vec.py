from collections import defaultdict
import gzip
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve
import difflib
import re
import numpy as n


class Word2Vec():

    def __init__(self, filepath, vocab_size=50000):
        self.words, self.embeddings = self.load_wordvec(filepath, vocab_size)
        # Mappings for O(1) retrieval:
        self.word2id = {word: idx for idx, word in enumerate(self.words)}
        self.id2word = {idx: word for idx, word in enumerate(self.words)}

    def load_wordvec(self, filepath, vocab_size):
        assert str(filepath).endswith('.gz')
        words = []
        embeddings = []
        with gzip.open(filepath, 'rt', encoding="utf8") as f:  # Read compressed file directly
            next(f)  # Skip header
            for i, line in enumerate(f):
                word, vec = line.split(' ', 1)
                words.append(word)
                embeddings.append(np.fromstring(vec, sep=' '))
                if i == (vocab_size - 1):
                    break
        print('Loaded %s pretrained word vectors' % (len(words)))
        return words, np.vstack(embeddings)

    def encode(self, word):
        # Returns the 1D embedding of a given word
        # return self.embeddings[self.word2id[word]]
        try:
            i = self.word2id[word]
            return self.embeddings[i]
        except:
            try:
                word = difflib.get_close_matches(word, self.words)[0]
                i = self.word2id[word]
            except:
                return np.zeros((300))
        return self.embeddings[i]

    def score(self, word1, word2):
        # Return the cosine similarity: use np.dot & np.linalg.norm
        code1 = self.encode(word1)
        code2 = self.encode(word2)
        return np.dot(code1, code2) / (np.linalg.norm(code1) * np.linalg.norm(code2))


class BagOfWords():
    def __init__(self, word2vec):
        self.word2vec = word2vec

    def build_idf(self, sentences):
        # build the idf dictionary: associate each word to its idf value
        # -> idf = {word: idf_value, ...}
        idf = {}
        N = len(sentences)

        # get number of documents containing each word
        for sentence in sentences:
            wordsList = re.sub("[^\w]", " ", sentence).split()
            for word in set(wordsList):
                idf[word] = idf.get(word, 0) + 1

        # transform to get idf value of each word
        for word in idf:
            idf[word] = np.log10(N / idf[word])
        return idf

    def encode(self, sentence, ag_sentence=True, padding=25, idf=None):

        # Takes a sentence as input, returns the sentence embedding
        wordsList = re.sub("[^\w]", " ", sentence).split()
        wordsVectors = [self.word2vec.encode(word) for word in wordsList]

        if ag_sentence == False:
            wordsVectors = wordsVectors[0:padding]
            wordsVectors = np.stack(wordsVectors, axis=0)
            wordsVectors = np.pad(wordsVectors, [(0, padding - len(wordsVectors)), (0, 0)], mode='constant')
            return wordsVectors

        if idf is None:
            # mean of word vectors
            return np.mean(wordsVectors, axis=0)
        else:
            # idf-weighted mean of word vectors
            weightedMean = 0
            sumIdf = 0
            for i, word in enumerate(wordsList):
                weightedMean += idf.get(word, 0) * wordsVectors[i]
                sumIdf += idf.get(word, 0)
            weightedMean = weightedMean / sumIdf
            return weightedMean

    def score(self, sentence1, sentence2, idf=None):
        # cosine similarity: use np.dot & np.linalg.norm
        code1 = self.encode(sentence1, idf)
        code2 = self.encode(sentence2, idf)
        return np.dot(code1, code2) / (np.linalg.norm(code1) * np.linalg.norm(code2))
