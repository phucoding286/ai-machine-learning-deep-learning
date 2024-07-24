class LogisticRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.weight = np.zeros((X.shape[1]))
        self.bias = np.zeros(1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predictions(self, x):
        z = np.dot(x, self.weight) + self.bias
        out = self.sigmoid(z)
        return out

    def training(self, epoch=20, lr=0.01, details=False, break_train=0.4):
        m = self.X.shape[0]
        
        for _ in range(epoch):
            y_pred = self.predictions(self.X)
            error = self.y - y_pred

            gradient_w = np.dot(self.X.T, error) / m
            gradient_b = np.mean(error)

            self.weight += lr * gradient_w
            self.bias += lr * gradient_b

            loss = np.mean(np.abs(error))
            if details:
                print(f"loss: {loss}")
            if loss < break_train:
                break
