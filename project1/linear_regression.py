import numpy as np

class LinearRegression():

    def __init__(self, learning_rate=0.001, epochs=1000):
        self.weights = None
        self.bias = None

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.residuals = []

    def fit(self, X, y):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """


        self.weights = np.array(0.0)
        self.bias = 0.0


        for _ in range(self.epochs):
            y_pred = self.predict(X)

            # Calculate gradients
            dw = (-2 / len(y)) * np.sum(X * (y - y_pred))
            db = (-2 / len(y)) * np.sum(y - y_pred)


            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        self.residuals = y - self.predict(X)


    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)

        Returns:
            A length m array of floats
        """
        X = np.array(X)
        return self.weights * X + self.bias