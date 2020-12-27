import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize_scalar


class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None,
                 random_state=None, **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        if trees_parameters is None:
            self.trees_parameters = dict()
        else:
            self.trees_parameters = dict(trees_parameters)
        self.n_estimators = n_estimators
        if feature_subsample_size is None:
            self.n_features = 1.
        elif feature_subsample_size <= 1:
            self.n_features = feature_subsample_size
        else:
            raise ValueError('feature_subsample_size cannot be greater than 1')
        self.trees = []
        self.trees_parameters['max_depth'] = max_depth
        self.trees_parameters['criterion'] = 'mse'
        # в случайном лесе в каждой вершине делается случайная выборка признаков:
        self.trees_parameters['splitter'] = 'random'
        if random_state is None:
            self.random_state = np.random.randint(0, 1000)
        else:
            self.random_state = random_state
        self.feat_inds = None

    def fit(self, X, y, X_val=None, y_val=None, return_log=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """

        feat_all_inds = np.arange(0, X.shape[1])
        feat_num = int(self.n_features * X.shape[1])
        if feat_num == 0:
            raise ValueError('inappropriate feature_subsample_size for the shape of X')
        self.feat_inds = np.zeros((self.n_estimators, feat_num), dtype=int)
        np.random.seed(self.random_state)
        if return_log:
            pred = 0
            log_loss = []
        for i in range(self.n_estimators):
            feat_cur_inds = np.random.choice(feat_all_inds, feat_num, replace=False)
            self.feat_inds[i, :] = np.array(feat_cur_inds)
            X_i = X[:, feat_cur_inds]
            tree_i = DecisionTreeRegressor(**self.trees_parameters)
            tree_i.fit(X_i, y)
            self.trees.append(tree_i)
            if return_log:
                pred += tree_i.predict(X_i)
                log_loss.append(round(mean_squared_error(pred / (i + 1), y)**0.5, 4))
        if return_log:
            return np.arange(1, self.n_estimators + 1), log_loss


    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pred = 0
        for i in range(self.n_estimators):
            pred += self.trees[i].predict(X[:, self.feat_inds[i, :]])
        pred /= self.n_estimators
        return pred


class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5,
                 feature_subsample_size=None, random_state=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use learning_rate * gamma instead of gamma
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        if trees_parameters is None:
            self.trees_parameters = dict()
        else:
            self.trees_parameters = dict(trees_parameters)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        if feature_subsample_size is None:
            self.n_features = None
        elif feature_subsample_size <= 1:
            self.n_features = feature_subsample_size
        else:
            raise ValueError('feature_subsample_size cannot be greater than 1')
        self.f_0 = None
        self.trees = []
        self.coefs = []
        self.trees_parameters['max_depth'] = max_depth
        self.trees_parameters['criterion'] = 'mse'
        # в случайном лесе в каждой вершине делается случайная выборка признаков:
        self.trees_parameters['splitter'] = 'random'
        if random_state is None:
            self.random_state = np.random.randint(0, 1000)
        else:
            self.random_state = random_state
        self.feat_inds = None

    def fit(self, X, y, X_val=None, y_val=None, return_log=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """

        def mse_coef(coef, f_pred, f_cur, y):
            return ((f_pred + coef * f_cur - y) ** 2).mean()

        if self.n_features is None:
            feat_num = X.shape[1]
        else:
            feat_num = int(self.n_features * X.shape[1])
        feat_all_inds = np.arange(0, X.shape[1])
        if feat_num == 0:
            raise ValueError('inappropriate feature_subsample_size for the shape of X')
        self.feat_inds = np.zeros((self.n_estimators, feat_num), dtype=int)

        # начальное приближение сделаем константным
        # среднее значение минимизирует MSE
        self.f_0 = y.mean()
        f_pred = self.f_0 * np.ones_like(y)
        n = y.shape[0]
        if return_log:
            log_loss = [round(mean_squared_error(f_pred, y)**0.5, 4)]
        np.random.seed(self.random_state)
        for i in range(self.n_estimators):
            feat_cur_inds = np.random.choice(feat_all_inds, feat_num, replace=False)
            self.feat_inds[i, :] = np.array(feat_cur_inds)
            X_i = X[:, feat_cur_inds]

            r_i = -2*(f_pred - y)/n
            tree_i = DecisionTreeRegressor(**self.trees_parameters)
            tree_i.fit(X_i, r_i)
            f_i = tree_i.predict(X_i)
            coef_i = self.learning_rate * minimize_scalar(mse_coef, args=(f_pred, f_i, y))['x']
            self.trees.append(tree_i)
            self.coefs.append(coef_i)
            f_pred += coef_i * f_i
            if return_log:
                log_loss.append(round(mean_squared_error(f_pred, y)**0.5, 4))
        if return_log:
            return np.arange(self.n_estimators + 1), log_loss

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pred = self.f_0
        for i in range(self.n_estimators):
            pred += self.coefs[i] * self.trees[i].predict(X[:, self.feat_inds[i, :]])
        return pred
