#!/usr/bin/env python
# coding: utf-8

# In[698]:


from distutils.command.upload import upload
import pandas as pd
import numpy as np
import json

import string

from tqdm import tqdm
tqdm.pandas()


# In[699]:


from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Embedding, MaxPooling1D, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, GRU
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils

import matplotlib.pyplot as plt

import http.server
import socketserver
import uploadserver
PORT = 8000
#get_ipython().run_line_magic('matplotlib', 'inline')


handler = http.server.SimpleHTTPRequestHandler

#with socketserver.TCPServer(("", PORT), handler) as httpd:
#    print("Server started at localhost:" + str(PORT))
#    httpd.serve_forever()

def learn():
# In[ ]:


    with open('./agora_hack_products/agora_hack_products.json', encoding='utf-8') as f:
       prdct = json.load(f)


    # In[ ]:


    prdct[0]


    # In[ ]:


    df = pd.DataFrame.from_dict(prdct, orient='columns')


    # In[ ]:


    df.info()


    # In[ ]:


    df.sample(3)


    # In[ ]:


    df[df['is_reference'] == True].count()


    # In[ ]:


    labels = df[df['is_reference'] == True]['reference_id'].count()


    # In[ ]:


    df.loc[(df['is_reference'] == True),'reference_id'] = df['product_id']


    # In[ ]:


    df[df['is_reference'] == True].sample(3)


    # In[ ]:


    df['reference_id'].value_counts().plot(kind = 'barh', figsize = (20, 20))


    # In[ ]:


    def jn(x):
        x = " ".join([ch for ch in x])
        x = str(x)
        return x


    # In[ ]:


    df['props_un'] = df['props'].apply(jn)


    # In[ ]:


    def jn_name(x):
        x = "".join([ch for ch in x])
        x = str(x)
        return x


    # In[ ]:


    df['props_un'] = df['props_un']+ ' ' +df['name'].apply(jn_name)


    # In[ ]:


    df.sample(3)


    # In[ ]:


    def rem_tab(x):
        x = x.replace("\t", " ")
        return x


    # In[ ]:


    df['props_un'] = df['props_un'].progress_apply(rem_tab)


    # In[ ]:


    df.sample(3)


    # In[3]:


    spec_chars = string.punctuation + '«'+ '»'+ '—'+ '"'+ '"'
    print(spec_chars)


    # In[4]:


    def rem_spec_chars(x):
        x = "".join([ch for ch in x if ch not in spec_chars])
        x = "".join([x.replace('\d+', '')])
        return x


    # In[5]:


    df['props_un'] = df['props_un'].progress_apply(rem_spec_chars)


    # In[ ]:


    df.sample()


    # In[ ]:


    def low(x):
      x = list(x.split())
      x = [w.lower() for w in x]
      return x


    # In[ ]:


    df['props_un'] = df['props_un'].progress_apply(low)


    # In[ ]:


    import nltk
    nltk.download('stopwords')


    # In[ ]:


    from nltk.corpus import stopwords
    russian_stopwords = stopwords.words("russian")


    # In[ ]:


    def stop_words(x):
      new_x = []
      for w in x:
        if w not in russian_stopwords:
            new_x.append(w)
      return new_x


    # In[ ]:


    df['props_un'] = df['props_un'].progress_apply(stop_words)


    # In[ ]:


    df.sample(3)


    # In[ ]:


    import pymorphy2 


    # In[ ]:


    morph = pymorphy2.MorphAnalyzer()


    # In[ ]:


    def lem(x):
        #x = list(x.split())
        x = [morph.parse(w)[0].normal_form for w in x]
        return x


    # In[ ]:


    df['props_un'] = df['props_un'].progress_apply(lem)


    # In[ ]:


    df.sample(3)


    # In[ ]:


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


    # In[ ]:


    pop_words = []
    for i in tqdm(unique_words.keys()):
        if unique_words[i] < 4:
            pass
        else:
            pop_words.append(i)


    # In[ ]:


    len(pop_words)


    # In[ ]:


    df['props_un_len'] = df['props_un'].apply(lambda x: len(x))


    # In[ ]:


    df['props_un_len'].value_counts(bins = 10).plot(kind = 'barh',figsize = (10, 5))


    # In[ ]:


    # Максимальное количество слов 
    num_words = len(pop_words)
    # Максимальная длина новости
    max_news_len = 67
    # Количество классов новостей
    nb_classes = 471


    # In[ ]:


    X = df['props_un']
    y = df['reference_id']


    # In[ ]:


    y = pd.get_dummies(y)


    # In[ ]:


    y.sample()


    # In[ ]:


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y, 
                                                        test_size=0.2)


    # In[ ]:


    y_train = np.array(y_train)
    y_test = np.array(y_test)


    # In[ ]:


    tokenizer = Tokenizer(num_words=num_words)


    # In[ ]:


    tokenizer.fit_on_texts(df['props_un'])


    # In[ ]:


    with open("./agora_hack_products/tokinaizer.json", "w") as outfile:
        json.dump(tokenizer.word_index, outfile)


    # In[ ]:


    train_sequences = tokenizer.texts_to_sequences(X_train)

    x_train = pad_sequences(train_sequences, maxlen=max_news_len)

    test_sequences = tokenizer.texts_to_sequences(X_test)

    x_test = pad_sequences(test_sequences, maxlen=max_news_len)


    # In[ ]:


    model_cnn = Sequential()
    model_cnn.add(Embedding(num_words, 128, input_length=max_news_len))
    model_cnn.add(Conv1D(1024, 5, padding='valid', activation='relu'))
    model_cnn.add(GlobalMaxPooling1D())
    model_cnn.add(Dense(512, activation='relu'))
    model_cnn.add(Dense(471, activation='softmax'))


    # In[ ]:


    model_cnn.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])


    # In[ ]:


    model_cnn.summary()


    # In[ ]:


    model_cnn_save_path = './model_weights/best_model_cnn.h5'
    checkpoint_callback_cnn = ModelCheckpoint(model_cnn_save_path, 
                                          monitor='val_accuracy',
                                          save_best_only=True,
                                          verbose=1)
    #model_cnn.save_model('./model_weights/best_model_cnn')

    # In[ ]:


    history_cnn = model_cnn.fit(x_train, 
                                y_train, 
                                epochs=10,
                                batch_size=64,
                                validation_split=0.02,
                                callbacks=[checkpoint_callback_cnn])


    # In[ ]:


    #plt.plot(history_cnn.history['accuracy'], 
    #         label='Доля верных ответов на обучающем наборе')
    #plt.plot(history_cnn.history['val_accuracy'], 
    #         label='Доля верных ответов на проверочном наборе')
    #plt.xlabel('Эпоха обучения')
    #plt.ylabel('Доля верных ответов')
    #plt.legend()
    #plt.show()


    # ## Cеть LSTM

    # In[ ]:


    #model_lstm = Sequential()
    #model_lstm.add(Embedding(num_words, 256, input_length=max_news_len))
    #model_lstm.add(LSTM(128))
    #model_lstm.add(Dense(471, activation='softmax'))


    # In[ ]:


    #model_lstm.compile(optimizer='adam', 
    #              loss='categorical_crossentropy', 
    #              metrics=['accuracy'])
    #

    # In[ ]:


    #model_lstm.summary()


    # In[ ]:


    #model_lstm_save_path = './model_weights/best_model_lstm.h5'
    #checkpoint_callback_lstm = ModelCheckpoint(model_lstm_save_path, 
    #                                      monitor='val_accuracy',
    #                                      save_best_only=True,
    #                                      verbose=1)


    # In[ ]:


    #history_lstm = model_lstm.fit(x_train, 
    #                              y_train, 
    #                              epochs=10,
    #                              batch_size=64,
    #                              validation_split=0.02,
    #                              callbacks=[checkpoint_callback_lstm],
    #                              use_multiprocessing=True)


    # In[ ]:


    #plt.plot(history_lstm.history['accuracy'], 
    #         label='Доля верных ответов на обучающем наборе')
    #plt.plot(history_lstm.history['val_accuracy'], 
    #         label='Доля верных ответов на проверочном наборе')
    #plt.xlabel('Эпоха обучения')
    #plt.ylabel('Доля верных ответов')
    #plt.legend()
    #plt.show()


    # In[ ]:


    #model_gru = Sequential()
    #model_gru.add(Embedding(num_words, 304, input_length=max_news_len))
    #model_gru.add(GRU(152))
    #model_gru.add(Dense(471, activation='softmax'))


    # In[ ]:


    #model_gru.compile(optimizer='adam', 
    #              loss='categorical_crossentropy', 
    #              metrics=['accuracy'])


    # In[ ]:


    #model_gru.summary()


    # In[ ]:


    #model_gru_save_path = './model_weights/best_model_gru.h5'
    #checkpoint_callback_gru = ModelCheckpoint(model_gru_save_path, 
    #                                      monitor='val_accuracy',
    #                                      save_best_only=True,
    #                                      verbose=1)


    # In[ ]:


    #history_gru = model_gru.fit(x_train, 
    #                              y_train, 
    #                              epochs=10,
    #                              batch_size=64,
    #                              validation_split=0.02,
    #                              callbacks=[checkpoint_callback_gru])


    # In[ ]:


    #plt.plot(history_gru.history['accuracy'], 
    #         label='Доля верных ответов на обучающем наборе')
    #plt.plot(history_gru.history['val_accuracy'], 
    #         label='Доля верных ответов на проверочном наборе')
    #plt.xlabel('Эпоха обучения')
    #plt.ylabel('Доля верных ответов')
    #plt.legend()
    #plt.show()


    # In[ ]:


    model_cnn.load_weights(model_cnn_save_path)


    # In[ ]:


    model_cnn.evaluate(x_test, y_test, verbose=1)


    # In[ ]:


    #model_lstm.load_weights(model_lstm_save_path)


    # In[ ]:


    #model_lstm.evaluate(x_test, y_test, verbose=1)


    # In[ ]:


    #model_gru.load_weights(model_gru_save_path)


    # In[ ]:


    #model_gru.evaluate(x_test, y_test, verbose=1)


    # In[ ]:


    y_test_pred_cnn = model_cnn.predict(x_test, verbose=1)
    #y_test_pred_gru = model_gru.predict(x_test, verbose=1)


    # In[ ]:


    test_pred = np.zeros(y_test_pred_cnn.shape)
    for i in tqdm(range(y_test_pred_cnn.shape[0])):
        for j in range(y_test_pred_cnn.shape[1]):
            test_pred[i,j] = max(y_test_pred_cnn[i,j])

    test_pred.shape


    # In[ ]:


    test_pred_class = []
    for i in range(len(test_pred)):
        index, max_value = max(enumerate(test_pred[i]), key=lambda i_v: i_v[1])
        test_pred_class.append((index))



def identify():
    t=Tokenizer()
    with open("./agora_hack_products/tokinaizer.json", encoding='utf-8') as f:
       t = json.load(f)
    
    #with open('./agora_hack_products/input.json', encoding='utf-8') as f:
    with open('./agora_hack_products/agora_hack_products.json', encoding='utf-8') as f:
       prdct = json.load(f)
    
    df = pd.DataFrame.from_dict(prdct, orient='columns')
    def jn(x):
        x = " ".join([ch for ch in x])
        x = str(x)
        return x




    df['props_un'] = df['props'].apply(jn)
    # Далее код разбора запроса
    model_cnn = Sequential()
    model_cnn_save_path = './model_weights/best_model_cnn.h5'
    
    model_cnn = Sequential()


    model_cnn.load_weights(model_cnn_save_path)
    #model_cnn.load_model(model_cnn_save_path)
    t.fit_on_texts(df['props_un'])
    x_test = df['props_un']
    y_test_pred_cnn = model_cnn.predict(x_test, verbose=1)
    test_pred = np.zeros(y_test_pred_cnn.shape)
    for i in tqdm(range(y_test_pred_cnn.shape[0])):
        for j in range(y_test_pred_cnn.shape[1]):
            test_pred[i,j] = max(y_test_pred_cnn[i,j])

    test_pred.shape
    test_pred_class = []
    test_pred_values = []
    for i in range(len(test_pred)):
        index, max_value = max(enumerate(test_pred[i]), key=lambda i_v: i_v[1])
        test_pred_class.append((index))
        test_pred_values.append((max_value))
    print(test_pred_class)

#learn()
identify()