import os
import numpy as np
import tensorflow as tf


def load_data():
    os.chdir("../data/txt/5.refined_rate")
    f_data = open("train_set_data.txt", 'r', encoding='utf-8')
    f_label = open("train_set_label.txt", "r", encoding='utf-8')

    data_lines = f_data.read().split()
    f_data.close()

    label_lines = f_label.read().split()
    f_label.close()

    train_data = np.array(data_lines, np.float)
    train_data = train_data.reshape(-1, 1000)

    train_label = np.array(label_lines, np.int)
    train_label = train_label.reshape(-1, 4)

    s = np.arange(train_data.shape[0])
    np.random.shuffle(s)

    train_data = train_data[s]
    train_label = train_label[s]

    print(train_data.shape)
    # print(train_data[:5])
    print(train_label.shape)
    # print(train_label[:5])

    f_data = open("test_set_data.txt", 'r', encoding='utf-8')
    f_label = open("test_set_label.txt", "r", encoding='utf-8')

    data_lines = f_data.read().split()
    f_data.close()

    label_lines = f_label.read().split()
    f_label.close()

    test_data = np.array(data_lines, np.float)
    test_data = test_data.reshape(-1, 1000)

    test_label = np.array(label_lines, np.int)
    test_label = test_label.reshape(-1, 4)

    os.chdir("../../../src")

    return train_data, train_label, test_data, test_label

