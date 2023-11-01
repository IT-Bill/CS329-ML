import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, x):
        distances = np.sum(np.power(self.X_train - x, 2), axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        return np.argmax(np.bincount(k_nearest_labels))

def train(n_neighbors, X, y):
    model = KNN(n_neighbors)
    model.fit(X, y)
    return model

if __name__ == "__main__":
    num_train = int(input())
    X_train, y_train = [], []
    for _ in range(num_train):
        data = list(map(float, input().strip().split()))
        X_train.append(data[1:])
        y_train.append(int(data[0]))

    X_train = np.array(X_train)
    y_train = np.array(y_train)


    model = {
        3: train(3, X_train, y_train),
        5: train(5, X_train, y_train),
        7: train(7, X_train, y_train),
    }

    num_test = int(input())
    for _ in range(num_test):
        data = list(map(float, input().strip().split()))
        neighbor = int(data[0])
        X_test = np.array(data[1:]).reshape(1, -1)

        print(model[neighbor].predict(X_test))
    
    print(X_train.tolist())
    print(y_train.tolist())
    print(X_test.tolist())

    
