from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, GRU


# сверточная нейронная сеть
class CNN_byRuslan:
    # Задаем гиперпараметры
    def __init__(self, num_words, max_news_len, model_cnn_save_path='./output/best_model_cnn.h5'):
        self.model_cnn = Sequential()
        self.checkpoint_callback_cnn = None
        self.num_words = num_words
        self.max_news_len = max_news_len
        self.model_cnn_save_path = model_cnn_save_path

    # Выполняет сборку модели
    def build(self):
        self.model_cnn.add(Embedding(self.num_words, 128, input_length=self.max_news_len))
        self.model_cnn.add(Conv1D(1024, 5, padding='valid', activation='relu'))
        self.model_cnn.add(GlobalMaxPooling1D())
        self.model_cnn.add(Dense(512, activation='relu'))
        self.model_cnn.add(Dense(471, activation='softmax'))
        self.model_cnn.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

    # обучение модели
    def forward(self, x_train, y_train):
        return self.model_cnn.fit(x_train,
                                  y_train,
                                  epochs=1,
                                  batch_size=64,
                                  validation_split=0.02)

    # Загрузка весов
    def load_weights(self, path):
        self.model_cnn.load_weights(path)

    # предсказание
    def predict(self, data):
        return self.model_cnn.predict(data, verbose=1)
