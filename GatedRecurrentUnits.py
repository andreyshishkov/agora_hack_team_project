from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, GRU

# GRU нейронная сеть
class GRU_byRuslan:
    # Задаем гиперпараметры
    def __init__(self, num_words, max_news_len, model_gru_save_path='./output/best_model_cnn.h5'):
        self.model_gru = Sequential()
        self.num_words = num_words
        self.max_news_len = max_news_len
        self.model_gru_save_path = model_gru_save_path

    # выполняем сборку модели
    def build(self):
        self.model_gru.add(Embedding(self.num_words, 304, input_length=self.max_news_len))
        self.model_gru.add(GRU(152))
        self.model_gru.add(Dense(471, activation='softmax'))

        self.model_gru.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])


    # Выполняем обучение модели
    def forward(self, x_train, y_train):
        return self.model_gru.fit(x_train,
                                  y_train,
                                  epochs=10,
                                  batch_size=64,
                                  validation_split=0.02)

    # загрузка весов
    def load_weights(self, path):
        self.model_gru.load_weights(path)

    # Выполнение предсказания
    def predict(self, data):
        return self.model_gru.predict(data, verbose=1)
