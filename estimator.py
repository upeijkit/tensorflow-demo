import numpy as np
import tensorflow as tf
# tf.estimator的使用练习，使用预定义的的训练模型或者自定义的训练模型

#定义模型训练函数，同时也定义了特征向量

def model_fn(features, labels, mode):
    #构建线性模型
    W = tf.get_variable("W",[1],dtype=tf.float64)
    b = tf.get_variable("b",[1],dtype=tf.float64)
    y = W * features['x'] + b
    #构建损失函数
    loss = tf.reduce_sum(tf.square(y - labels))
    #训练模型子图
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step,1))
    #通过EstimatorSpec指定我们的训练子图积极损失模型
    return tf.estimator.EstimatorSpec(mode=mode,predictions=y,loss=loss,train_op=train)

#创建自定义的模型向量
estimator = tf.estimator.Estimator(model_fn=model_fn)

#创建一个特征向量列表，该特征列表里只有一个特征向量，为实数向量，只有一个元素的数组，名为X
#feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

#创建一个linearRegressor训练器，并传入特征向量列表
#estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

#保存训练用的数据
x_train = np.array([1.,2.,3.,6.,8.])
y_train = np.array([4.8,8.5,10.4,21.0,25.3])

#保存评估用的数据
x_eval = np.array([2.,5.,7.,9.])
y_eval = np.array([7.6,17.2,23.6,28.8])

#用训练数据创建一个输入模型，用来进行后面的模型训练
#第一个参数作为线性回归模型的输入数据
#第二个参数用来作为损失模型的输入
#第三个参数表示每批训练数据的个数
#第四个参数为迭代次数，将训练集的所有数据训练一遍为一次迭代
#第五个参数为取训练数据是顺序取还是随机取
train_input_fn = tf.estimator.inputs.numpy_input_fn({"x" : x_train}, y_train,batch_size=2, num_epochs=None,shuffle=True)
#再用训练数据创建一个输入模型，用来进行后面的模型评估
train_input_fn_2 = tf.estimator.inputs.numpy_input_fn({"x":x_train},y_train,batch_size=2,num_epochs=1000,shuffle=False)
#再用评估数据创建一个输入模型，用来进行后面的模型评估
eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval},y_eval,batch_size=2,num_epochs=1000,shuffle=False)
#使用训练数据训练1000次
estimator.train(input_fn=train_input_fn,steps=1000)
#使用训练数据评估一下模型，目的是查看训练结果
train_metrics = estimator.evaluate(input_fn=train_input_fn_2)
print("train metrics: %r" % train_metrics)
#使用评估数据评估一下模型，目的是验证模型的泛化能力
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("eval metrics: %s"  % eval_metrics)