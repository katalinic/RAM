import pickle
import gzip
import numpy as np
import RAM

def load_data(path):
    f = gzip.open(path + 'mnist.pkl.gz','rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return training_data, validation_data, test_data

def vectorised_result(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def get_all_data(path):
    
    train_d, _, test_d = load_data(path)
    train = train_d[0].reshape(-1,28,28)
    y = train_d[1]
    target = np.array([vectorised_result(y) for y in y]).reshape(-1,10)

    test = test_d[0].reshape(-1,28,28)
    test_y = test_d[1]
    test_target = np.array([vectorised_result(y) for y in test_y]).reshape(-1,10)

    return train, target, test, test_target

def train():

    train, target, test, test_target = get_all_data('your_path')

    RAM = RAM.RAM()
    RAM.train_model(train, target)
    RAM.test_model(test, test_target)

if __name__=="__main__":
    train()
