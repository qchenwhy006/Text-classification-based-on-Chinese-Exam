import tensorflow as tf, numpy as np
from models.base import ClassifierBase


class FastText(ClassifierBase):
    def __init__(self,
                 maxlen=260,
                 max_features=200000,
                 embedding_dims=100,
                 class_num=2,
                 class_type='binary',
                 batch_size=64,
                 ngram_range=None):
        super(FastText, self).__init__(class_num=class_num, class_type=class_type)

        # parameters
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.class_type = class_type
        self.batch_size = batch_size
        self.ngram_range = ngram_range
        self.tokenizer = None
        self.y_encoder = None
        self.ngram_indice = None

        # layer
        self.embedding = None
        self.pooling = None
        self.outputs = None

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam()

    def __create_ngram_set(self, input_list, ngram_value=2):
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    def __add_ngram(self, sequences, token_indice, ngram_range=2):
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for ngram_value in range(2, ngram_range + 1):
                for i in range(len(new_list) - ngram_value + 1):
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in token_indice:
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)
        return np.array(new_sequences)

    def __initialize_ngram_indices(self, x):
        ngram_set = set()
        for input_list in x:
            for i in range(2, self.ngram_range + 1):
                set_of_ngram = self.__create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        start_index = len(self.tokenizer.word_index) + 1
        self.ngram_indice = {v: k + start_index for k, v in enumerate(ngram_set) if
                             (k + start_index) < self.max_features}

    def __fit_on_x(self, x):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.max_features, oov_token='<UNK>')
        self.tokenizer.fit_on_texts(x)

        if self.ngram_range is not None and self.ngram_range > 1:
            self.__initialize_ngram_indices(x)

    def transform_on_x(self, x):
        x = self.tokenizer.texts_to_sequences(x)
        if self.ngram_range is not None:
            x = self.__add_ngram(x, self.ngram_indice, self.ngram_range)
        return tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=self.maxlen, padding='post', truncating='post')

    def __fit_and_transform_on_x(self, x):
        self.__fit_on_x(x)
        return self.transform_on_x(x)

    def __train_one_step(self, x, y):
        with tf.GradientTape() as tape:
            hidden_state = self.embedding(x)
            hidden_state = self.pooling(hidden_state)
            output = self.outputs(hidden_state)
            batch_loss = self.loss_function(y, output)
            variables = self.embedding.trainable_variables + self.pooling.trainable_variables + self.outputs.trainable_variables
            gradients = tape.gradient(batch_loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            return batch_loss

    def fit(self, x, y, epochs=5):
        x = self.__fit_and_transform_on_x(x)
        y = self.fit_and_transform_on_y(y)

        # initialize layers
        self.embedding = tf.keras.layers.Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        if self.class_type == 'multi-class':
            self.outputs = tf.keras.layers.Dense(self.class_num, activation='softmax')
        else:
            self.outputs = tf.keras.layers.Dense(self.class_num, activation='sigmoid')

        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(x) // self.batch_size):
                start_idx = self.batch_size * i
                end_idx = min(len(x), (i + 1) * self.batch_size)
                batch_loss = self.__train_one_step(x[start_idx: end_idx], y[start_idx: end_idx])
                total_loss += batch_loss

            print('Epoch {} Loss: {}'.format(epoch, total_loss / (len(x) // self.batch_size)))

    @tf.function
    def __call__(self, x):
        hidden_state = self.embedding(x)
        hidden_state = self.pooling(hidden_state)
        output = self.outputs(hidden_state)
        return output

    def predict_proba(self, x):
        x = self.transform_on_x(x)
        hidden_state = self.embedding(x)
        hidden_state = self.pooling(hidden_state)
        output = self.outputs(hidden_state)
        return output.numpy()

    def predict(self, x):
        proba = self.predict_proba(x)
        if self.class_type == 'multi-class':
            trans_proba = np.argmax(proba, axis=1)
            result = np.zeros((len(x), self.class_num))
            for idx, pos in enumerate(trans_proba):
                result[idx][pos] = 1
            return result
        else:
            return (proba > 0.5).astype(int)