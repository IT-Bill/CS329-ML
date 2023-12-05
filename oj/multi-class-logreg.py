import numpy as np

def softmax(z):
    # Compute softmax values for each set of scores in z
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / e_z.sum(axis=1, keepdims=True)

def train_softmax_regression(features, targets, epochs, lr):
    # Initialize weights (including bias term)
    n_samples, n_features = features.shape
    n_classes = targets.shape[1]
    weights = np.zeros((n_features + 1, n_classes))

    # Add bias term to features
    features = np.hstack([features, np.ones((n_samples, 1))])

    # Training loop
    for epoch in range(epochs):
        # Compute predictions using softmax
        predictions = softmax(np.dot(features, weights))

        # Compute the gradient
        gradient = np.dot(features.T, (predictions - targets)) / n_samples

        # Update weights
        weights -= lr * gradient

    return weights

def main():
    # Read input parameters
    
    N, D, C, E, L = map(str, input().split())
    N, E, L = int(N), int(E), float(L)

    # Read feature samples
    features = np.array([list(map(float, input().split())) for _ in range(N)])

    # Read target samples
    targets = np.array([list(map(float, input().split())) for _ in range(N)])

    # Train the model
    model_weights = train_softmax_regression(features, targets, E, L)

    # Reshape and output the model weights
    model_weights_reshaped = model_weights.reshape(-1)
    for weight in model_weights_reshaped:
        print("%.3f" % weight)

if __name__ == "__main__":
    main()
