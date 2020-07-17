import tensorflow as tf, numpy as np
from sklearn.metrics.classification import f1_score
from sklearn.model_selection import train_test_split
from models.base import ClassifierBase


class TextCNN(ClassifierBase):
    def __init__(self,
                 maxlen=260,
                 max_features=200000,
                 embedding_dims=100,
                 class_num=2,
                 class_type='binary',
                 batch_size=64,
                 kernal_sizes=[2, 3, 4],
                 filter_num=16,
                 conv_stride=1,
                 ealystop_metric='macro_f1',
                 patience=3,
                 save_path='checkpoint/textcnn',
                 word_vector_dict=None,
                 fine_tuning=True):
        super(TextCNN, self).__init__(class_num, class_type)

        # params
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.class_type = class_type
        self.batch_size = batch_size
        self.kernal_sizes = kernal_sizes
        self.filter_num = filter_num
        self.conv_stride = conv_stride
        self.earlystop_metric = ealystop_metric
        self.patience = patience
        self.word_vector_dict = word_vector_dict
        self.fine_tuning = fine_tuning
        self.save_path = save_path

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam()
        self.y_encoder = None
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.max_features, oov_token='<UNK>')

        # layers
        self.embedding = tf.keras.layers.Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        self.conv_list = []
        self.pooling_list = []
        for size in self.kernal_sizes:
            self.conv_list.append(
                tf.keras.layers.Conv1D(filters=self.filter_num, kernel_size=size, strides=self.conv_stride))
            self.pooling_list.append(tf.keras.layers.GlobalMaxPooling1D())
        self.concat = tf.keras.layers.Concatenate()
        if self.class_type == 'multi-class':
            self.outputs = tf.keras.layers.Dense(self.class_num, activation='softmax')
        else:
            self.outputs = tf.keras.layers.Dense(self.class_num, activation='sigmoid')

    def __fit_on_x(self, x):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.max_features, oov_token='<UNK>')
        self.tokenizer.fit_on_texts(x)

    def transform_on_x(self, x):
        x = self.tokenizer.texts_to_sequences(x)
        return tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=self.maxlen, padding='post', truncating='post')

    def __fit_and_transform_on_x(self, x):
        self.__fit_on_x(x)
        return self.transform_on_x(x)

    def __create_embedding(self):
        if self.word_vector_dict is None:
            return tf.keras.layers.Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        else:
            embedding_matrix = np.zeros((self.max_features, self.embedding_dims))
            for i in range(1, min(self.max_features, len(self.tokenizer.index_word) + 1)):
                embedding_matrix[i] = self.word_vector_dict[self.tokenizer.index_word[i]]
            return tf.keras.layers.Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen,
                                             weights=[embedding_matrix], trainable=self.fine_tuning)

    def __train_one_step(self, x, y):
        with tf.GradientTape() as tape:
            output = self.__forward(x)
            batch_loss = self.loss_function(y, output)
            variables = self.embedding.trainable_variables
            for idx in range(len(self.kernal_sizes)):
                variables += self.conv_list[idx].trainable_variables
                variables += self.pooling_list[idx].trainable_variables
            variables += (self.concat.trainable_variables + self.outputs.trainable_variables)
            gradients = tape.gradient(batch_loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            return batch_loss

    def __forward(self, x):
        embeddings = self.embedding(x)
        branch_list = []
        for idx in range(len(self.kernal_sizes)):
            hidden_state = self.conv_list[idx](embeddings)
            hidden_state = self.pooling_list[idx](hidden_state)
            branch_list.append(hidden_state)
        hidden_state = tf.keras.layers.Concatenate()(branch_list) if len(self.kernal_sizes) > 1 else branch_list[0]
        output = self.outputs(hidden_state)
        return output

    def change_proba_to_digits(self, proba):
        if self.class_type == 'multi-class':
            trans_proba = np.argmax(proba, axis=1)
            result = np.zeros((len(proba), self.class_num))
            for idx, pos in enumerate(trans_proba):
                result[idx][pos] = 1
            return result
        else:
            return (proba > 0.5).astype(int)

    def fit(self, x, y, validation_data=None, validation_split=0.1, epochs=5):
        self.checkpoint = tf.train.Checkpoint(model=self)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.save_path, max_to_keep=1)

        x = self.__fit_and_transform_on_x(x)
        y = self.fit_and_transform_on_y(y)
        if validation_data is None:
            train_x, train_y, val_x, val_y = train_test_split(x, y, test_size=validation_split, random_state=2020)
        else:
            train_x, train_y = x, y
            val_x = self.transform_on_x(validation_data[0])
            val_y = self.transform_on_y(validation_data[1])

        max_score = None
        best_epoch = None
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(train_x) // self.batch_size):
                start_idx = self.batch_size * i
                end_idx = min(len(train_x), (i + 1) * self.batch_size)
                batch_loss = self.__train_one_step(train_x[start_idx: end_idx], train_y[start_idx: end_idx])
                total_loss += batch_loss
                print('Epoch {} Batch {} Batch Loss: {}'.format(epoch, i, batch_loss))

            val_y_proba = self.__forward(val_x).numpy()
            val_y_pred = self.change_proba_to_digits(val_y_proba)
            curr_macro_score = f1_score(val_y, val_y_pred, average='macro')
            curr_micro_score = f1_score(val_y, val_y_pred, average='micro')
            print('Epoch {} Loss: {} Macro F1: {} Micro F1: {}'.format(epoch, total_loss / (len(x) // self.batch_size),
                                                                       curr_macro_score, curr_micro_score))
            if self.earlystop_metric == 'macro_f1':
                curr_score = curr_macro_score
            elif self.earlystop_metric == 'micro_f1':
                curr_score = curr_macro_score
            else:
                raise Exception('This metric is not supported now.')

            if max_score is None or curr_score > max_score:
                self.checkpoint_manager.save()
                max_score = curr_score
                best_epoch = epoch
            elif (epoch - best_epoch) >= self.patience:
                print(
                    'Early stopped at epoch {}, since macro f1 score does not improve from {}'.format(epoch, max_score))
                break

    @tf.function
    def __call__(self, x):
        output = self.__forward(x)
        return output

    def predict_proba(self, x):
        x = self.transform_on_x(x)
        output = self.__forward(x)
        return output.numpy()

    def predict(self, x):
        proba = self.predict_proba(x)
        return self.change_proba_to_digits(proba)

