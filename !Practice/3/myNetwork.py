import numpy as np

def relu(x):
    return np.maximum(0, x)

def learn():

    X = np.array([[0.5,-0.2]])

    Weight1 = np.array([[0.1,0.2,-0.3],
                   [0.4,-0.5,0.6]])
    bias1 = np.array([[0.1,-0.1,0.2]])

    # input layer
    z1 = np.dot(X, Weight1) + bias1
    print(f"Inpur layer:", z1) # [[0.07 0.1  -0.1 ]]

    activation1 = relu(z1) # [0.07, 0.1, 0.0]
    print(f"Output layer:", activation1)


    Weight2 = np.array([[0.3], 
                   [-0.2], 
                   [0.5]])
    bias2 = np.array([[0.1]])

    # output layer
    z2 = np.dot(activation1, Weight2) + bias2

    # MSE (loss function)
    y_pred = z2
    y_true = 0.7
    loss = np.mean((y_pred - y_true) ** 2)
    print("Loss:", loss)

if __name__ == "__main__":
    print("My Network")
    learn()

