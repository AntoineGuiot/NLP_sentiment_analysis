import numpy as np
import pandas as pd
import word2vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pathlib import Path
from urllib.request import urlretrieve
import os
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate
import scipy

current_folder = os.getcwd()


class Classifier:
    """The Classifier"""

    #############################################     
    def remove_stopwords(self, sentence):
        stop_words = set(stopwords.words('english'))
        stop_words.remove("not")
        stop_words.remove("no")
        sentence = sentence.lower()
        sentence_tok = word_tokenize(sentence)
        sentence_f = ""
        for i in range(len(sentence_tok)):
            w = sentence_tok[i]
            if w not in stop_words:
                if i == len(sentence_tok) - 1:
                    sentence_f += w
                else:
                    sentence_f += w + " "
        if len(sentence_f) < 2:
            sentence_f = sentence
        return sentence_f

    def read_data(self, source, train_data=True):
        df = pd.read_csv(source, sep='\t', header=None)

        df.columns = ["polarity", "aspect_category", "target_term", "character_offset", "sentence"]
        df["label"] = df["polarity"].apply(
            lambda x: 1 if x == "positive" else (0 if x == "neutral" else -1))

        ### Formating output
        label = to_categorical(df['label'] + 1)

        # remove target
        sentence_red = [0] * len(df)
        for i in range(len(df)):
            sentence_red[i] = df["sentence"][i][:int(df["character_offset"][i].split(":")[0])] + \
                              df["sentence"][i][int(
                                  df["character_offset"][i].split(":")[1]):]

        df["sentence_red"] = sentence_red
        # remove stopwords
        # df["sentence_red"] = df["sentence_red"].apply(lambda x: self.remove_stopwords(x))
        # word2vec embeddings
        PATH_TO_DATA = Path('../data')
        en_embeddings_path = PATH_TO_DATA / 'cc.en.300.vec.gz'
        print(en_embeddings_path)
        if not en_embeddings_path.exists():
            urlretrieve('https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz', en_embeddings_path)

        w2vec = word2vec.Word2Vec(en_embeddings_path, vocab_size=50000)
        sentence2vec = word2vec.BagOfWords(w2vec)
        # Each word in a sentence will be encoded with its corresponding vector (size 300) in the embedding layer,
        # Each sentence contains N word -> N vector of size 300. In order to have the same input size for each sentence
        # we will pad the sequence of words with zeros vector until 100 words (for sentences with more than 100 words:
        # we will truncate the sentence at 100 words. (meaning we could possibly loss information)
        # If needed, this parameter could be set as more than 100

        sentences = [sentence2vec.encode(df["sentence"][i], ag_sentence=False, padding=100) for i in
                     range(len(df["sentence"]))]

        #### Transform as array
        sentences = np.stack(sentences)

        #### Encoding categorie (oneHotEncoding):
        if train_data == True:
            self.enc = OneHotEncoder(handle_unknown='ignore')
            self.enc.fit(df['aspect_category'].values.reshape(-1, 1))
        categories = self.enc.transform(df['aspect_category'].values.reshape(-1, 1))

        return (sentences, categories, label)

    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        train_sentences, categories_train, label_train = self.read_data(trainfile, True)

        # dev_sentences, categories_dev, label_dev = self.read_data(devtestfile, False)

        # Building NN architecture:
        inputA = Input(shape=(100, 300))
        inputB = Input(shape=(12,))
        y = Dense((16), activation='relu')(inputB)

        x = Dense((16), activation="relu")(inputA)
        x = LSTM(16, return_sequences=False, input_shape=(50, 300), go_backwards=True)(x)
        z = concatenate([x, y])
        output = Dense(3, activation='softmax')(z)

        self.model = Model(inputs=[inputA, inputB], outputs=output)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model.summary()  # show the summary of this model in logs
        # trainig :

        if type(train_sentences) == scipy.sparse.csc.csc_matrix:
            train_sentences = train_sentences.toarray()

        if type(categories_train) == scipy.sparse.csc.csc_matrix:
            categories_train = categories_train.toarray()

        self.model.fit((train_sentences, categories_train),
                       label_train,
                       epochs=50,
                       batch_size=124,
                       # validation_data=((dev_sentences, categories_dev), label_dev),
                       verbose=1)

    def predict(self, source):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """

        sentences, categorie, _ = self.read_data(source, False)

        if type(sentences) == scipy.sparse.csc.csc_matrix:
            sentences = sentences.toarray()

        if type(categorie) == scipy.sparse.csc.csc_matrix:
            categorie = categorie.toarray()

        predictions = self.model.predict((sentences, categorie))
        predictions = np.argmax(predictions, axis=1) - 1
        predictions = predictions.tolist()
        polarity = []
        for p in predictions:
            if p == 1:
                polarity.append("positive")
            elif p == 0:
                polarity.append("neutral")
            else:
                polarity.append("negative")
        return (polarity)
