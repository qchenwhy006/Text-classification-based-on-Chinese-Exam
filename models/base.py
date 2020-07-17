import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MultiLabelBinarizer



class ClassifierBase(tf.keras.Model):
    def __init__(self, class_num=2, class_type='binary'):
        super(ClassifierBase, self).__init__()
        self.class_num = class_num
        self.class_type = class_type
        self.y_encoder = None

    def loss_function(self, y_true, y_pred):
        if self.class_type == 'multi-class':
            loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='none')
        else:
            loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')

        loss_ = loss_object(y_true, y_pred)
        return tf.reduce_mean(loss_, axis=0)

    def fit_on_y(self, y):
        if self.class_type == 'binary':
            # binary
            self.y_encoder = LabelEncoder()
        elif 'class' in self.class_type:
            # multi-class
            self.y_encoder = LabelBinarizer()
        elif 'label' in self.class_type:
            # multi-label
            self.y_encoder = MultiLabelBinarizer()
            y = [l.split(' ') for l in y]
        else:
            raise Exception("'classification_type' must be one of 'binary', 'multi-class', 'multi-label'")
        # encode y
        self.y_encoder.fit(y)

    def transform_on_y(self, y):
        if 'label' in self.class_type:
            y = [l.split(' ') for l in y]
        return self.y_encoder.transform(y)

    def fit_and_transform_on_y(self, y):
        self.fit_on_y(y)
        return self.transform_on_y(y)

    def predict(self, x):
        raise NotImplementedError

    def predict_proba(self, x):
        raise NotImplementedError