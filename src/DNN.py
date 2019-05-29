import tensorflow as tf
import matplotlib.pyplot as plt
import data2numpy
import numpy as np


def cal_predict(label, predict):
    korean_acc = 0
    math_acc = 0
    eng_acc = 0
    sci_acc = 0
    korean_num = 0
    math_num = 0
    eng_num = 0
    sci_num = 0

    for i in range(predict.shape[0]):
        if label[i][0] == 1:
            korean_num += 1
            if predict[i].argmax() == 0:
                korean_acc += 1

        elif label[i][1] == 1:
            math_num += 1
            if predict[i].argmax() == 1:
                math_acc += 1

        elif label[i][2] == 1:
            eng_num += 1
            if predict[i].argmax() == 2:
                eng_acc += 1

        elif label[i][3] == 1:
            sci_num += 1
            if predict[i].argmax() == 3:
                sci_acc += 1

    print("국어 정확도: ", korean_acc, "/", korean_num, "=", korean_acc / korean_num)
    print("수학 정확도: ", math_acc, "/", math_num, "=", math_acc / math_num)
    print("영어 정확도: ", eng_acc, "/", eng_num, "=", eng_acc / eng_num)
    print("과학 정확도: ", sci_acc, "/", sci_num, "=", sci_acc / sci_num)
    print("-------------------------------")


def DNN():
    batch_size = 10

    train_data, train_label, test_data, test_label = data2numpy.load_data()

    n = train_data.shape[0]
    n2 = train_data.shape[1]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(n2, input_shape=(n2,), activation='relu'))
    model.add(tf.keras.layers.Dense(n2, input_shape=(n2,), activation='relu'))
    model.add(tf.keras.layers.Dense(n2, input_shape=(n2,), activation='relu'))
    model.add(tf.keras.layers.Dense(n2, input_shape=(n2,), activation='relu'))
    model.add(tf.keras.layers.Dense(n2, input_shape=(n2,), activation='relu'))
    model.add(tf.keras.layers.Dense(n2, input_shape=(n2,), activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='sigmoid'))
    model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

    # for i in range(epoch):
    #     batch_xs = train_data[i:i+batch_size, :]
    #     batch_ys = train_label[i:i+batch_size, :]
    hist = model.fit(train_data, train_label, validation_data=(test_data, test_label),
                     epochs=3)

    trainset_predict = model.predict(train_data)
    testset_predict = model.predict(test_data)

    print("훈련셋 정확도")
    cal_predict(train_label, trainset_predict)
    print("\n테스트셋 정확도")
    cal_predict(test_label, testset_predict)

    plt.figure(figsize=(12, 8))
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.legend(['loss', 'val_loss', 'acc', 'val_acc'])
    plt.show()

    return model

if __name__ == "__main__":
    DNN()
