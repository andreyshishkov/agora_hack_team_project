import json
import string
import nltk
import numpy as np
import pandas as pd
import pymorphy2
from nltk.corpus import stopwords
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from convolutionalNeuralNetwork import CNN_byRuslan
from GatedRecurrentUnits import GRU_byRuslan

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


# на вход получает 2 предсказания из CNN & GRU
# test, y
# write2File=True - автоматическая запись предсказания в файл, в таком случае нужно передать. По умолчание result.json
# write2File=False - вернет предсказание
def predict2json(y_test_pred_cnn, y_test_pred_gru, test, y, path='result.json', write2File=True):
    test_pred = np.zeros(y_test_pred_cnn.shape)
    for i in tqdm(range(y_test_pred_cnn.shape[0])):
        for j in range(y_test_pred_cnn.shape[1]):
            test_pred[i, j] = max(y_test_pred_cnn[i, j], y_test_pred_gru[i, j])

    test_pred_class = []
    for i in range(len(test_pred)):
        index, max_value = max(enumerate(test_pred[i]), key=lambda i_v: i_v[1])
        test_pred_class.append(index)

    test['reference_id'] = 0

    for i in tqdm(test.index):
        test['reference_id'].iloc[i] = test_pred_class[i]

    test['reference_id'] = test['reference_id'].apply(lambda x: y.columns[x])
    out = test[['id', 'reference_id']].to_json(orient='records')

    if write2File:
        with open(path, 'w') as f:
            f.write(out)
    else:
        return out


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

    cnn = CNN_byRuslan(num_words=num_words,
                       max_news_len=max_news_len)
    cnn.build() # сборка модели
    #history_cnn = cnn.forward(x_train, y_train) # обучение

    # Cеть GRU
    print("Сеть GRU: ")
    gru_model = GRU_byRuslan(num_words=num_words,
                            max_news_len=max_news_len)
    gru_model.build()
    #history_gru = gru.forward(x_train, y_train)

    with open('agora_hack_products/test_request.json', encoding='utf-8') as f:
        tst = json.load(f)

    # Создание DataFrame из словаря
    test = pd.DataFrame.from_dict(tst, orient='columns')
    # Предобработка данных
    prepr(test)

    test_2_sequences = tokenizer.texts_to_sequences(test['props'])

    x_test_2 = pad_sequences(test_2_sequences, maxlen=max_news_len)

    # загрузка весов
    cnn.load_weights('./output/best_model_cnn.h5')

    gru_model.load_weights('./output/best_model_gru.h5')

    # выполняем предсказание
    y_test_pred_cnn = cnn.predict(x_test_2)
    y_test_pred_gru = gru_model.predict(x_test_2)

    predict2json(y_test_pred_cnn, y_test_pred_gru, test, y)

if __name__ == '__main__':
    main()
