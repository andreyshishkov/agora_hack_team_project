import json
import string
import nltk
import numpy as np
import pandas as pd
import pymorphy2
from nltk.corpus import stopwords
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, GRU
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

tqdm.pandas()

# Preprocessing texts
def prepr(df):
    # т.к. описание в формате list с несколькими данными объединим в один список

    nltk.download('stopwords')
    russian_stopwords = stopwords.words("russian")
    morph = pymorphy2.MorphAnalyzer()

    print("Начинаем обработку текстовых данных!")
    df['props_un'] = df['props'].apply(lambda x: str(" ".join([ch for ch in x])))

    # добавим имя продукта для обработки
    df['props_un'] = df['props_un'] + ' ' + df['name'].apply(lambda x: str("".join([ch for ch in x])))

    # избавляемся от табуляции
    df['props_un'] = df['props_un'].progress_apply(lambda x: x.replace("\t", " "))

    # избавляемся от знаков препинания
    def rem_spec_chars(x):
        spec_chars = string.punctuation + '«' + '»' + '—' + '"' + '"'
        x = "".join([ch for ch in x if ch not in spec_chars])
        x = "".join([x.replace('\d+', '')])
        return x

    df['props_un'] = df['props_un'].progress_apply(rem_spec_chars)

    # переводим всё в нижний регистр
    df['props_un'] = df['props_un'].progress_apply(lambda x: [w.lower() for w in x.split()])
    # удаляем стоп слова
    df['props_un'] = df['props_un'].progress_apply(lambda x: [w for w in x if w not in russian_stopwords])
    # лемматизируем текст
    df['props_un'] = df['props_un'].progress_apply(lambda x: [morph.parse(w)[0].normal_form for w in x])
    print("Обработка текстовых данных закончилась")


# создадим словарь уникальных слов для токенизации

def count_words(df):
    unique_words = {}
    for i in tqdm(df.index):
        for j in set(df['props_un'].loc[i]):
            if len(j) == 1:
                pass
            else:
                if j in unique_words.keys():
                    unique_words[j] += 1
                else:
                    unique_words[j] = 1
    pop_words = []
    for i in tqdm(unique_words.keys()):
        if unique_words[i] < 4:
            pass
        elif unique_words[i] > len(unique_words) * 0.9:
            pass
        else:
            pop_words.append(i)
    return pop_words


# сверточная нейронная сеть
class CNN_byRuslan:
    def __init__(self, num_words, max_news_len, model_cnn_save_path='./output/best_model_cnn.h5'):
        self.num_words = num_words
        self.max_news_len = max_news_len
        self.model_cnn_save_path = model_cnn_save_path

    def forward(self, x_train, y_train):
        model_cnn = Sequential()
        model_cnn.add(Embedding(self.num_words, 128, input_length=self.max_news_len))
        model_cnn.add(Conv1D(1024, 5, padding='valid', activation='relu'))
        model_cnn.add(GlobalMaxPooling1D())
        model_cnn.add(Dense(512, activation='relu'))
        model_cnn.add(Dense(471, activation='softmax'))
        model_cnn.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        checkpoint_callback_cnn = ModelCheckpoint(self.model_cnn_save_path,
                                                  monitor='val_accuracy',
                                                  save_best_only=True,
                                                  verbose=1)
        return model_cnn.fit(x_train,
                             y_train,
                             epochs=10,
                             batch_size=64,
                             validation_split=0.02,
                             callbacks=[checkpoint_callback_cnn])


class GRU_byRuslan:
    def __init__(self, num_words, max_news_len, model_gru_save_path='./output/best_model_cnn.h5'):
        self.num_words = num_words
        self.max_news_len = max_news_len
        self.model_gru_save_path = model_gru_save_path

    def forward(self, x_train, y_train):
        model_gru = Sequential()
        model_gru.add(Embedding(self.num_words, 304, input_length=self.max_news_len))
        model_gru.add(GRU(152))
        model_gru.add(Dense(471, activation='softmax'))
        model_gru.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        checkpoint_callback_gru = ModelCheckpoint(self.model_gru_save_path,
                                                  monitor='val_accuracy',
                                                  save_best_only=True,
                                                  verbose=1)
        return model_gru.fit(x_train,
                             y_train,
                             epochs=10,
                             batch_size=64,
                             validation_split=0.02,
                             callbacks=[checkpoint_callback_gru])


def main():
    df = pd.read_json('agora_hack_products/agora_hack_products.json')
    labels = df[df['is_reference'] == True]['product_id'].count()

    df.loc[(df['is_reference'] == True), 'reference_id'] = df['product_id']

    prepr(df)
    # Максимальное количество слов
    num_words = len(count_words(df))
    # Максимальная длина новости
    max_news_len = 67
    # Количество классов новостей
    nb_classes = labels

    # разделим трейн и тест
    X = df['props_un']
    y = df['reference_id']

    y = pd.get_dummies(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=0.2)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # делаем токенизатор на списке всех слов и сохраним его в отдельном файле
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(df['props_un'])

    with open("./output/tokinaizer.json", "w") as outfile:
        json.dump(tokenizer.word_index, outfile)

    # токенизируем трейн и тест
    train_sequences = tokenizer.texts_to_sequences(X_train)
    x_train = pad_sequences(train_sequences, maxlen=max_news_len)
    test_sequences = tokenizer.texts_to_sequences(X_test)
    x_test = pad_sequences(test_sequences, maxlen=max_news_len)

    # Сверточная нейронная сеть
    print("Светроточная нейронная сеть: ")
    history_cnn = CNN_byRuslan(num_words=num_words,
                               max_news_len=max_news_len,
                               model_cnn_save_path='./output/best_model_cnn_2.h5').forward(x_train, y_train)

    # Cеть GRU
    print("Сеть GRU: ")
    history_gru = CNN_byRuslan(num_words=num_words,
                               max_news_len=max_news_len,
                               model_cnn_save_path='./output/best_model_gru_2.h5').forward(x_train, y_train)


if __name__ == '__main__':
    main()
