import numpy as np
import matplotlib.pyplot as plt

# 对数几率函数, logistic function
def logistic(x):
    return 1.0/(1 + np.exp(-x))

class MyLogisticRegression:
    def fit(self, x, y, params):
        sample_num, attribute_num = np.shape(x)
        learn_rate = params['learn_rate']
        apoch = params['apoch']
        w = np.ones((attribute_num, 1))
        # traing process
        for k in range(apoch):
            # 不同的学习策略进行学习
            if params['train_type']=='gradDescent':
                output = logistic(x * w)
                error = y - output
                w = w + learn_rate * np.transpose(x) * error
            elif params['train_type'] =='stocGradDescent':
                for i in range(sample_num):
                    output = logistic(x[i,:] * w)
                    error = y[i,:] - output
                    w = w + learn_rate * x[i, :].transpose() * error
            elif params['train_type'] =='smoothStocGradDescent':
                dataIndex = list(range(sample_num))
                for i in range(sample_num):
                    learn_rate = 4.0/(1.0 + k + i) +0.01  # learn_rate随着时间的后移和轮数的增加逐渐变小
                    randIndex = int(np.random.uniform(0, len(dataIndex))) # 随机抽取样本进行学习，防止过拟合
                    output = logistic(x[dataIndex[randIndex], :] * w)
                    error = train_y[dataIndex[randIndex], 0] - output
                    w = w + learn_rate * x[dataIndex[randIndex], :].transpose() * error
                    del (dataIndex[randIndex])  # during one interation, delete the optimized sample
            else:
                raise NameError('Not support optimize method type!')

        self.w = w

    def predict(self, x):
        sample_num, attribute_num = np.shape(x)
        right_num = 0
        ans = []
        for i in range(sample_num):
            z = logistic(x[i, :] * self.w)[0, 0]
            predict = z > 0.5
            ans.append(predict)
        return ans

    def predict_test(self, x, y):
        sample_num, attribute_num = np.shape(x)
        right_num = 0
        ans = self.predict(x)
        for i in range(sample_num):
            if ans[i] == bool(y[i, 0]):
                right_num = right_num + 1
        acc = 1.0 * right_num / sample_num
        return ans,acc

    def paint_classify_result(self, x, y):
        sample_num, attribute_num = np.shape(x)
        if attribute_num != 3 :
            print("Sorry,this program cannot deal with the high dimension data for the dimension of your data is not 2")
            return 0
        # paint each samples int the pic
        for i in range(sample_num):
            if int(y[i]) == 0 :
                plt.plot(x[i, 1], x[i, 2], 'ob')
            elif int(y[i]) !=0 :
                plt.plot(x[i, 1], x[i, 2], 'xg')

        # paint the classify line of the model
        # the min point (min_x1,min_x2)
        min_x1 = min(x[:,1])[0,0]
        min_x2 = float(-self.w[0] - self.w[1]*min_x1)/self.w[2]
        # the max point (max_x1,max_x2)
        max_x1 = max(x[:,1])[0,0]
        max_x2 = float(-self.w[0] - self.w[1]*min_x2)/self.w[2]
        plt.plot([min_x1, min_x2], [max_x1, max_x2], '--r')
        plt.xlabel('ATTR1')
        plt.ylabel('ATTR2')
        plt.show()

    def fix_data(self, x, y):
        # x is a 2-dimension array
        # y is a 1-dimension list
        sample_num = np.shape(x)[0]
        add_one = np.ones(sample_num).reshape(sample_num,1)
        # merge the x and add_one
        # 在所有数据列的前面加一列1，表示表达式中的常数项，简化方程
        x = np.hstack((add_one,x))
        # np.mat() can trans train_x into the type of matrix,
        # so we can calculate the x[i,:]*self.w and it results a num instead of a 2-dimension array
        x = np.mat(x)
        y = np.mat(y).transpose()
        return x, y
def loadData():
    train_x = []
    train_y = []
    fileIn = open('./testSet.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        train_x.append([float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return train_x,train_y

train_x, train_y = loadData()
model = MyLogisticRegression()
train_x, train_y = model.fix_data(train_x, train_y)
test_x = train_x
test_y = train_y
# |- train_type             -|
# -----------------------------
# |- gradDescent            -|
# |- stocGradDescent        -|
# |- smoothStocGradDescent  -|
opts = {'learn_rate': 0.01, 'apoch': 50, 'train_type': 'smoothStocGradDescent'}
model = MyLogisticRegression()
model.fit(train_x, train_y, opts)

ans, accuracy = model.predict_test(test_x, test_y)
print(accuracy)
print(train_x.shape[0])
model.paint_classify_result(test_x, test_y)
