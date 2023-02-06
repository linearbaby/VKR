from sklearn.svm import LinearSVC
import numpy as np

def aquire_model():
    data = np.random.randn(100, 2)
    data[:50] += (2, 5)
    data[50:] += (-3, -2)
    
    y = np.zeros(100)
    y[:50] = 1

    model = LinearSVC()
    model.fit(data, y)
    return model

