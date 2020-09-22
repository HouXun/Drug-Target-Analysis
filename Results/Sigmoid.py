import numpy as np
from matplotlib import pyplot as plt


# ②　定义sigmoid()函数。
def sigmoid(x):
    return 1/(1 + np.exp(-x))


# ③　使用np.linspace生成5000个从-10到10的数据，并赋值给变量x_data。
x_data = np.linspace(-10,10,5000)
# ④　调用sigmoid函数，函数接收参数为x_data。
y = sigmoid(x_data)

# ⑤　使用matplotlib画图显示sigmoid函数曲线。
# 曲线
plt.plot(x_data,y)
# 分界点
plt.scatter(0,0.5,marker='*',c='r')
# 展示
plt.savefig('Sigmoid.jpg')
plt.show()
