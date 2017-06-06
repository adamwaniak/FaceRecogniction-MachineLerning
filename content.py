# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 3: Regresja logistyczna
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------
import functools

import numpy as np
import time


def sigmoid(x):
    '''
    :param x: wektor wejsciowych wartosci Nx1
    :return: wektor wyjściowych wartości funkcji sigmoidalnej dla wejścia x, Nx1
    '''

    return np.divide(np.ones(x.shape), 1 + np.exp(np.negative(x)))


def logistic_cost_function(w, x_train, y_train):
    '''
    :param w: parametry modelu Mx1
    :param x_train: ciag treningowy - wejscia NxM
    :param y_train: ciag treningowy - wyjscia Nx1
    :return: funkcja zwraca krotke (val, grad), gdzie val oznacza wartosc funkcji logistycznej, a grad jej gradient po w
    '''
    auxArray = x_train @ w
    sigmoidArray = sigmoid(auxArray)

    val = np.negative(np.divide(np.sum(y_train * auxArray - np.log(np.add(np.exp(auxArray), 1))), y_train.shape[0]))
    grad = np.divide((x_train.transpose() @ (np.subtract(sigmoidArray, y_train))), y_train.shape[0])

    return val, grad


def gradient_descent(obj_fun, w0, epochs, eta):
    '''
    :param obj_fun: funkcja celu, ktora ma byc optymalizowana. Wywolanie val,grad = obj_fun(w).
    :param w0: punkt startowy Mx1
    :param epochs: liczba epok / iteracji algorytmu
    :param eta: krok uczenia
    :return: funkcja wykonuje optymalizacje metoda gradientu prostego dla funkcji obj_fun. Zwraca krotke (w,func_values),
    gdzie w oznacza znaleziony optymalny punkt w, a func_valus jest wektorem wartosci funkcji [epochs x 1] we wszystkich krokach algorytmu
    '''
    func_values = list()
    w = w0
    for k in range(epochs):
        val, grad = obj_fun(w)
        w = w + eta * (-grad)
        func_values.append(val)

    val, _ = obj_fun(w)
    func_values.append(val)
    del func_values[0]
    return w, np.reshape(np.array(func_values), (epochs, 1))


def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    '''
    :param obj_fun: funkcja celu, ktora ma byc optymalizowana. Wywolanie val,grad = obj_fun(w,x,y), gdzie x,y oznaczaja podane
    podzbiory zbioru treningowego (mini-batche)
    :param x_train: dane treningowe wejsciowe NxM
    :param y_train: dane treningowe wyjsciowe Nx1
    :param w0: punkt startowy Mx1
    :param epochs: liczba epok
    :param eta: krok uczenia
    :param mini_batch: wielkosc mini-batcha
    :return: funkcja wykonuje optymalizacje metoda stochastycznego gradientu prostego dla funkcji obj_fun. Zwraca krotke (w,func_values),
    gdzie w oznacza znaleziony optymalny punkt w, a func_values jest wektorem wartosci funkcji [epochs x 1] we wszystkich krokach algorytmu. Wartosci
    funkcji do func_values sa wyliczane dla calego zbioru treningowego!
    '''

    def iterate_minibatches(x_train, y_train, batchsize):
        assert x_train.shape[0] == y_train.shape[0]

        for start_idx in range(0, x_train.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield x_train[excerpt], y_train[excerpt]

    w = w0
    func_values = list()

    for k in range(epochs):
        for batch in (iterate_minibatches(x_train, y_train, mini_batch)):
            x_batch, y_batch = batch
            val, grad = obj_fun(w, x_batch, y_batch)
            w = w + eta * (-grad)
        val, grad = obj_fun(w, x_train, y_train)
        func_values.append(val)

    return w, np.reshape(np.array(func_values), (epochs, 1))


def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    '''
    :param w: parametry modelu Mx1
    :param x_train: ciag treningowy - wejscia NxM
    :param y_train: ciag treningowy - wyjscia Nx1
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (val, grad), gdzie val oznacza wartosc funkcji logistycznej z regularyzacja l2,
    a grad jej gradient po w
    '''
    auxArray = x_train @ w
    sigmoidArray = sigmoid(auxArray)

    val = np.negative(np.divide(np.sum(y_train * auxArray - np.log(np.add(np.exp(auxArray), 1))), y_train.shape[0])) + (
                                                                                                                           regularization_lambda / 2) * (
                                                                                                                           np.linalg.norm(
                                                                                                                               np.delete(
                                                                                                                                   w,
                                                                                                                                   0)) ** 2)
    wz = np.copy(w)
    wz[0] = 0

    grad = np.divide((x_train.transpose() @ (np.subtract(sigmoidArray, y_train))),
                     y_train.shape[0]) + regularization_lambda * wz

    return val, grad

    pass


def prediction(x, w, theta):
    '''
    :param x: macierz obserwacji NxM
    :param w: wektor parametrow modelu Mx1
    :param theta: prog klasyfikacji z przedzialu [0,1]
    :return: funkcja wylicza wektor y o wymiarach Nx1. Wektor zawiera wartosci etykiet ze zbioru {0,1} dla obserwacji z x
     bazujac na modelu z parametrami w oraz progu klasyfikacji theta
    '''

    return np.where(sigmoid(x @ w) >= theta, 1, 0)


def f_measure(y_true, y_pred):
    '''
    :param y_true: wektor rzeczywistych etykiet Nx1
    :param y_pred: wektor etykiet przewidzianych przed model Nx1
    :return: funkcja wylicza wartosc miary F
    '''
    TP = np.sum(np.bitwise_and(y_true, y_pred))
    FPandFN = np.sum(np.bitwise_xor(y_true, y_pred))
    return 2 * TP / (2 * TP + FPandFN)


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    '''
    :param x_train: ciag treningowy wejsciowy NxM
    :param y_train: ciag treningowy wyjsciowy Nx1
    :param x_val: ciag walidacyjny wejsciowy Nval x M
    :param y_val: ciag walidacyjny wyjsciowy Nval x 1
    :param w0: wektor poczatkowych wartosci parametrow
    :param epochs: liczba epok dla SGD
    :param eta: krok uczenia
    :param mini_batch: wielkosc mini batcha
    :param lambdas: lista wartosci parametru regularyzacji lambda, ktore maja byc sprawdzone
    :param thetas: lista wartosci progow klasyfikacji theta, ktore maja byc sprawdzone
    :return: funckja wykonuje selekcje modelu. Zwraca krotke (regularization_lambda, theta, w, F), gdzie regularization_lambda
    to najlpszy parametr regularyzacji, theta to najlepszy prog klasyfikacji, a w to najlepszy wektor parametrow modelu.
    Dodatkowo funkcja zwraca macierz F, ktora zawiera wartosci miary F dla wszystkich par (lambda, theta). Do uczenia nalezy
    korzystac z algorytmu SGD oraz kryterium uczenia z regularyzacja l2.
    '''
    F = np.zeros((len(lambdas), len(thetas)))
    W = list()
    for lambda_index in range(len(lambdas)):
        w, _ = stochastic_gradient_descent(
            functools.partial(regularized_logistic_cost_function, regularization_lambda=lambdas[lambda_index]), x_train,
            y_train, w0, epochs, eta, mini_batch)
        W.append(w)

        for theta_index in range(len(thetas)):
            y_pred = prediction(x_val, w, thetas[theta_index])
            F[lambda_index, theta_index] = f_measure(y_val, y_pred)

    index = np.unravel_index(F.argmax(), F.shape)

    best_lambda = lambdas[index[0]]
    best_theta = thetas[index[1]]
    best_w = W[index[0]]

    return best_lambda, best_theta, best_w, F
