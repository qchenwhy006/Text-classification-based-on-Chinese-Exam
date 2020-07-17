import math
import numpy as np
import tensorflow as tf
from models.base import ClassifierBase
from transformers import TFBertModel, BertTokenizer, create_optimizer


class BERTClassifier(ClassifierBase):
    def __init__(self, class_num=2, class_type='binary', model_name='bert-base-chinese', max_length=512, batch_size=4,
                 output_dropout=0.2):
        super(BERTClassifier, self).__init__(class_num=class_num, class_type=class_type)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.output_dropout = output_dropout
        self.y_encoder = None

        # initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            self.model_name,
            do_lower_case=True,
            padding_side='right')

        # initialize layers
        self.bert_layer = TFBertModel.from_pretrained(
            self.model_name,
            max_length=self.max_length,
            finetuning_task=True,
            output_hidden_states=False,
            output_attentions=False)
        self.dropout_layer = tf.keras.layers.Dropout(self.output_dropout)
        if self.class_type == 'multi-class':
            self.output_layer = tf.keras.layers.Dense(self.class_num, activation='softmax')
        else:
            self.output_layer = tf.keras.layers.Dense(self.class_num, activation='sigmoid')

    def change_proba_to_digits(self, proba):
        if self.class_type == 'multi-class':
            trans_proba = np.argmax(proba, axis=1)
            result = np.zeros((len(proba), self.class_num))
            for idx, pos in enumerate(trans_proba):
                result[idx][pos] = 1
            return result
        else:
            return (proba > 0.5).astype(int)

    def __transform_x(self, x):
        input_ids = np.array([self.tokenizer.encode(text=item, add_special_tokens=True, max_length=self.max_length,
                                                    truncation_strategy='longest_first', pad_to_max_length=True,
                                                    return_tensors=None) for item in x], dtype=np.int64)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).astype(np.int64)
        return input_ids, attention_mask

    def __forward(self, x, attention_mask):
        hidden_state = self.bert_layer(x, attention_mask=attention_mask)
        hidden_state = self.dropout_layer(hidden_state[1])
        output = self.output_layer(hidden_state)
        return output

    def __train_one_step(self, x, attention_mask, y, optimizer):
        with tf.GradientTape() as tape:
            output = self.__forward(x, attention_mask)
            batch_loss = self.loss_function(y, output)
            variables = self.bert_layer.trainable_variables + self.dropout_layer.trainable_variables + self.output_layer.trainable_variables
            gradients = tape.gradient(batch_loss, variables)
            optimizer.apply_gradients(zip(gradients, variables), 1.0)
            return batch_loss

    def fit(self, x, y, epochs=2):
        input_ids, attention_mask = self.__transform_x(x)
        y = self.fit_and_transform_on_y(y)

        steps_per_epoch = math.ceil(len(x) / self.batch_size)
        warmup_steps = steps_per_epoch // 3
        total_steps = steps_per_epoch - warmup_steps
        optimizer = create_optimizer(init_lr=2e-5, num_train_steps=total_steps, num_warmup_steps=warmup_steps)

        for epoch in range(epochs):
            total_loss = 0

            for i in range(steps_per_epoch):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(x))
                batch_input_ids = input_ids[start_idx: end_idx]
                batch_attention_mask = attention_mask[start_idx: end_idx]
                batch_y = y[start_idx: end_idx]
                batch_loss = self.__train_one_step(batch_input_ids, batch_attention_mask, batch_y, optimizer)
                total_loss += batch_loss
                print('Epoch {} Batch {} Batch Loss: {}'.format(epoch, i, batch_loss))
            print('Epoch {} Loss: {}'.format(epoch, total_loss / steps_per_epoch))

    def predict_proba(self, x):
        input_ids, attention_mask = self.__transform_x(x)
        output = self.__forward(input_ids, attention_mask)
        return output.numpy()

    def predict(self, x):
        proba = self.predict_proba(x)
        return self.change_proba_to_digits(proba)

    def predict_in_batch(self, x, batch_size=40):
        y_pred = None
        steps = math.ceil(len(x) / batch_size)
        for step in range(steps):
            start_idx = step * batch_size
            end_idx = min((step + 1) * batch_size, len(x))
            cur_y_pred = self.predict(x[start_idx: end_idx])
            if y_pred is None:
                y_pred = cur_y_pred
            else:
                y_pred = np.concatenate([y_pred, cur_y_pred])
        return y_pred