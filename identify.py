import pandas as pd
import numpy as np
import json

import string

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")

import pymorphy2 
morph = pymorphy2.MorphAnalyzer()

from tqdm import tqdm
tqdm.pandas()

import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, MaxPooling1D, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, GRU
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils

import matplotlib.pyplot as plt

def identify():
    DIR = './'

    with open(DIR + 'agora_hack_products/agora_hack_products.json', encoding='utf-8') as f:
       prdct = json.load(f)

    df = pd.DataFrame.from_dict(prdct, orient='columns')

    labels = df[df['is_reference'] == True]['product_id'].count()

    df.loc[(df['is_reference'] == True),'reference_id'] = df['product_id']

    def prep(df):

        # т.к. описание в формате list с несколькими данными объединим в один список 

        def jn(x):
            x = " ".join([ch for ch in x])
            x = str(x)
            return x

        df['props_un'] = df['props'].apply(jn)

        # добавим имя продукта для обработки

        def jn_name(x):
            x = "".join([ch for ch in x])
            x = str(x)
            return x

        df['props_un'] = df['props_un']+ ' ' +df['name'].apply(jn_name)

        # избавляемся от табуляции

        def rem_tab(x):
            x = x.replace("\t", " ")
            return x

        df['props_un'] = df['props_un'].progress_apply(rem_tab)

        # избавляемся от знаков препинания

        spec_chars = string.punctuation + '«'+ '»'+ '—'+ '"'+ '"'
        print(spec_chars)

        def rem_spec_chars(x):
            x = "".join([ch for ch in x if ch not in spec_chars])
            x = "".join([x.replace('\d+', '')])
            return x

        df['props_un'] = df['props_un'].progress_apply(rem_spec_chars)

        # переводим всё в нижний регистр

        def low(x):
            x = list(x.split())
            x = [w.lower() for w in x]
            return x

        df['props_un'] = df['props_un'].progress_apply(low)

        # удаляем стоп слова

        def stop_words(x):
            new_x = []
            for w in x:
                if w not in russian_stopwords:
                    new_x.append(w)
            return new_x

        df['props_un'] = df['props_un'].progress_apply(stop_words)

        # лемматизируем текст

        def lem(x):
            #x = list(x.split())
            x = [morph.parse(w)[0].normal_form for w in x]
            return x

        df['props_un'] = df['props_un'].progress_apply(lem)

    prep(df)

    # создадим словарь уникальных слов для токенизации

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
        elif unique_words[i] > len(unique_words)*0.9:
            pass
        else:
            pop_words.append(i)

    # Максимальное количество слов 
    num_words = len(pop_words)
    # Максимальная длина новости
    max_news_len = 67
    # Количество классов новостей
    nb_classes = labels

    # разделим трейн и тест

    X = df['props_un']
    y = df['reference_id']

    y = pd.get_dummies(y)

    from sklearn.model_selection import train_test_split
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

    model_cnn = Sequential()
    model_cnn.add(Embedding(num_words, 128, input_length=max_news_len))
    model_cnn.add(Conv1D(1024, 5, padding='valid', activation='relu'))
    model_cnn.add(GlobalMaxPooling1D())
    model_cnn.add(Dense(512, activation='relu'))
    model_cnn.add(Dense(471, activation='softmax'))

    model_cnn.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])



    history_cnn = model_cnn.fit(x_train, 
                                y_train, 
                                epochs=1,
                                batch_size=64,
                                validation_split=0.02)

    # Cеть GRU

    model_gru = Sequential()
    model_gru.add(Embedding(num_words, 304, input_length=max_news_len))
    model_gru.add(GRU(152))
    model_gru.add(Dense(471, activation='softmax'))

    model_gru.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])


    history_gru = model_gru.fit(x_train, 
                                  y_train, 
                                  epochs=1,
                                  batch_size=64,
                                  validation_split=0.02)

    with open(DIR + 'agora_hack_products/test_request.json', encoding='utf-8') as f:
       tst = json.load(f)

    test = pd.DataFrame.from_dict(tst, orient='columns')

    prep(test)

    test_2_sequences = tokenizer.texts_to_sequences(test['props'])

    x_test_2 = pad_sequences(test_2_sequences, maxlen=max_news_len)

    model_cnn.load_weights('./output/best_model_cnn.h5')
    model_gru.load_weights('./output/best_model_gru.h5')

    y_test_pred_cnn = model_cnn.predict(x_test_2, verbose=1)
    y_test_pred_gru = model_gru.predict(x_test_2, verbose=1)

    test_pred = np.zeros(y_test_pred_cnn.shape)
    for i in tqdm(range(y_test_pred_cnn.shape[0])):
        for j in range(y_test_pred_cnn.shape[1]):
            test_pred[i,j] = max(y_test_pred_cnn[i,j], y_test_pred_gru[i,j])

    test_pred_class = []
    for i in range(len(test_pred)):
        index, max_value = max(enumerate(test_pred[i]), key=lambda i_v: i_v[1])
        test_pred_class.append((index))

    test['reference_id'] = 0

    for i in tqdm(test.index):
        test['reference_id'].iloc[i] = test_pred_class[i]

    test['reference_id'] = test['reference_id'].apply(lambda x: y.columns[x])

    out = test[['id', 'reference_id']].to_json(orient='records')

    #with open('./output/test_answer.json', 'w') as f:
    #    f.write(out)
    return out