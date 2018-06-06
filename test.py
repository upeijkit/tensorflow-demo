import tensorflow as tf
#用于练习TENSORFLOW基本API的使用
'''
t0 = tf.constant(3, dtype=tf.int32)
t1 = tf.constant([3., 4.1, 5.2], dtype=tf.float32)
t2 = tf.constant([['apple','orange'],['potato','tomato']], dtype=tf.string)
t3 = tf.constant([[[5],[6],[7]],[[4],[3],[2]]])

sess = tf.Session()

print(t0)
print(t1)
print(t2)
print(t3)
print(sess.run(t0))
print(sess.run(t1))
print(sess.run(t2))
print(sess.run(t3))

'''

'''
node1 = tf.constant(3.2)
node2 = tf.constant(4.8)
adder = node1 + node2
print(adder)
sess = tf.Session()
print(sess.run(adder))
'''
'''
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print(a)
print(b)
print(adder_node)
sess = tf.Session()
print(sess.run(adder_node,{a:3,b:4.5}))
print(sess.run(adder_node,{a:[1,3],b:[2,4]}))
add_and_triple = adder_node *3
print(sess.run(add_and_triple, {a:3,b:4.5}))
'''

W = tf.Variable([.1], dtype=tf.float32)
b = tf.Variable([-.1],dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W*x + b

y = tf.placeholder(tf.float32)

loss = tf.reduce_sum(tf.square(linear_model - y))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
#print(sess.run(w))
#print(sess.run(linear_model, {x:[1,2,3,6,8]}))
#print(sess.run(loss,{x:[1,2,3,6,8], y:[4.8,8.5,10.4,21.0,25.3]}))
#fixW = tf.assign(W,[2.])
#fixb = tf.assign(b,[1.])
#sess.run([fixW, fixb])
#print(sess.run(loss,{x:[1,2,3,6,8], y:[4.8,8.5,10.4,21.0,25.3]}))


optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

x_train = [1,2,3,6,8]
y_train = [4.8,8.5,10.4,21.0,25.3]

for i in range(10000):
    sess.run(train,{x: x_train, y: y_train })

print('W: %s b: %s loss: %s' % (sess.run(W), sess.run(b), sess.run(loss, {x: x_train,y: y_train})))


