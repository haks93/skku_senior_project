import DNN
import demo_refine as df

model = DNN.DNN()

while True:
    try:
        inputString = input()
        if inputString == "exit":
            break

        arr = df.refine(inputString)
        data = df.refine_2(1000, 100, arr)

        predict = model.predict(data)
        #print(predict)
        print("==================result==================")
        for i in range(predict.shape[0]):
            if predict[i].argmax() == 0:
                print("영어")

            if predict[i].argmax() == 1:
                print("국어")

            if predict[i].argmax() == 2:
                print("수학")

            if predict[i].argmax() == 3:
                print("과학")
        print("==========================================\n\n")
    except:
        continue




