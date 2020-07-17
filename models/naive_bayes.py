import numpy as np
from sklearn.preprocessing import LabelEncoder


class MultinomialNaiveBayes(object):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        """

        :param alpha: Additive smoothing parameter.
        If alpha = 1, then it is Laplace smoothing, else it is Lidstone smoothing.
        :param fit_prior: Whether to learn class prior probabilities or not.
        :param class_prior: Prior probabilities of the classes. If none, we will not use custom class prior.
        """
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.py_cache = None
        self.pxy_cache = None
        self.encoder = LabelEncoder()

    def fit(self, X, y):
        """

        :param X: Training vectors.
        :param y: Target values.
        :return: A fitted MNB model.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        y = self.encoder.fit_transform(y)

        self.py_cache = self.__get_y_dist(y)
        self.pxy_cache = self.__get_xy_dist(X, y)

        return self

    def predict_log_proba(self, X):

        X = np.asarray(X)
        return np.dot(X, self.pxy_cache.transpose()) + self.py_cache

    def predict_proba(self, X):

        return np.exp(self.predict_log_proba(X))

    def predict(self, X):

        log_proba = self.predict_log_proba(X)
        return self.encoder.inverse_transform(np.argmax(log_proba, axis=1))

    def __get_y_dist(self, y):
        if not self.fit_prior:
            return np.log(np.ones(len(self.encoder.classes_))) - np.log(len(self.encoder.classes_))
        elif self.class_prior is None:
            result = np.zeros(len(self.encoder.classes_))
            for i in range(len(self.encoder.classes_)):
                result[i] = np.log(len(y[y == i])) - np.log(len(y))
            return result
        else:
            return np.log(np.asarray(self.class_prior))

    def __get_xy_dist(self, X, y):
        result = []
        for i in range(len(self.encoder.classes_)):
            cur_X = X[y == i]
            temp = cur_X.sum(axis=0) + self.alpha
            result.append(np.log(temp) - np.log(temp.sum()))
        return np.asarray(result)