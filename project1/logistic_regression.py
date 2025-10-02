import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _compute_loss(self, y, y_pred):
        return -y*np.log(y_pred) - (1-y)*np.log(1-y_pred)

    def compute_gradients(self, x, y, y_pred):
        m = len(y)
        grad_w = (1/m) * np.dot(x.T, (y_pred - y))
        grad_b = (1/m) * np.sum(y_pred - y)
        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

    def accuracy(self, true_values, predictions):
        return np.mean(true_values == predictions)
    
    def fit(self, x, y):
        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for _ in range(self.epochs):
            lin_model = np.matmul(self.weights, x.transpose()) + self.bias
        
            y_pred = self._sigmoid(lin_model)
            grad_w, grad_b = self.compute_gradients(x, y, y_pred)
            self.update_parameters(grad_w, grad_b)

            loss = self._compute_loss(y, y_pred)
            pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]
            self.train_accuracies.append(self.accuracy(y, pred_to_class))
            self.losses.append(loss)

    def predict(self, x):
        lin_model = np.matmul(x, self.weights) + self.bias
        y_pred = self._sigmoid(lin_model)
        return [1 if _y > 0.5 else 0 for _y in y_pred]
    
    def predict_proba(self, x):
        lin_model = np.matmul(x, self.weights) + self.bias
        y_pred = self._sigmoid(lin_model)
        
        prob_class_0 = 1 - y_pred
        prob_class_1 = y_pred
        return np.column_stack([prob_class_0, prob_class_1])