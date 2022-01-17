import glob

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn.linear_model import SGDOneClassSVM
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors


class NoveltyDetector:
    def __init__(self, verb):
        self.verbose = verb
        print("verbose: {}".format(self.verbose))
        self.data_test = None
        self.data_train = None
        self.data_test_scaled = None
        self.data_train_scaled = None
        self.model = None
        self.predictions = None
        self.prediction_scores = None

    def __data_load(self, data_path):
        
        if self.verbose:
            print("Reading from: {}".format(data_path))
            
        file_list = glob.glob(data_path + '/*.npz')
        data = []
        for file in file_list:
            x = np.load(file)['yf']
            data.append(x)
            
        if self.verbose:
            print("Read {} files".format(len(file_list)))

        return data

    def __scale(self, train_data, test_data):
        if self.verbose:
            print("Scaling data...")

        scaler = StandardScaler()
        self.data_train_scaled = scaler.fit_transform(self.data_train)
        self.data_test_scaled = scaler.transform(self.data_test)

    def load_train(self, path):
        if self.verbose:
            print("Loading train data...")

        self.data_train = np.array(self.__data_load(path))

    def load_test(self, path):
        if self.verbose:
            print("Loading test data...")

        self.data_test = np.array(self.__data_load(path))

    def train(self, model):
        if self.verbose:
            print("Training ")

        match model:
            case "OneCSVM":
                if self.verbose:
                    print("OneClassSVM...")

                self.model = svm.OneClassSVM(nu=0.01, kernel="rbf")
            case "SGDOneCSVM":
                if self.verbose:
                    print("SGDOneClassSVM...")

                self.model = svm.SGDOneClassSVM()

        self.__scale(self.data_train, self.data_test)
        self.model.fit(self.data_train_scaled)

    def predict(self):
        if self.verbose:
            print("Predicting...")

        self.predictions = self.model.predict(self.data_test_scaled)
        self.prediction_scores = self.model.score_samples(self.data_test_scaled)

        unique, counts = np.unique(self.predictions, return_counts=True)
        print(np.asarray((unique, counts)).T)

    def visualize(self):
        self.__scale(self.data_train, self.data_test)
        X = np.concatenate([self.data_train, self.data_test], axis=0)
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(X)
        plt.scatter(X_embedded[:len(self.data_train), 0], X_embedded[:len(self.data_train), 1], color='blue', label='train data')
        plt.scatter(X_embedded[-len(self.data_test):, 0], X_embedded[-len(self.data_test):, 1], color='red', label='test data')
        plt.legend()
        plt.show()

    def OptimizeNu(self):
        x = 0.01
        step = 0.01
        min = 100000
        val = 0

        while 1:
            x += step
            self.model.fit(self.data_train)
            pred = self.model.predict(self.data_test)
            unique, counts = np.unique(pred, return_counts=True)

            if unique[0] == -1:
                if counts[0] < min:
                    min = counts[0]
                    val = x

            if x > 0.98:
                print("min: ", val)
                break

def main():
    data_path_train = "C:/Users/320121275/PycharmProjects/pythonProject1/data_set_train/"
    data_path_test = "C:/Users/320121275/PycharmProjects/pythonProject1/data_set_test/"
    # data_path_test = "C:/Users/320121275/PycharmProjects/pythonProject1/data_set_test1/"

    model = "OneCSVM"
    # model = "SGDOneCSVM"
    print_verbose = 1

    noveltyDetector = NoveltyDetector(print_verbose)
    noveltyDetector.load_train(data_path_train)
    noveltyDetector.load_test(data_path_test)
    noveltyDetector.train(model)
    noveltyDetector.predict()
    noveltyDetector.visualize()


if __name__ == "__main__":
    main()