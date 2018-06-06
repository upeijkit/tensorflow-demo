import tensorflow as tf

W = tf.Variable([0], dtype=tf.float32, name='W')
b = tf.Variable([0], dtype=tf.float32, name='b')

x = tf.placeholder(tf.float32,name='x')
y = tf.placeholder(tf.float32,name='y')

linear_model = W * x + b

with tf.name_scope("loss-model"):
    loss = tf.reduce_sum(tf.square(linear_model-y))
    tf.summary.scalar("loss",loss)

optimizer = tf.train.GradientDescentOptimizer(0.001)

train = optimizer.minimize(loss)

x_train = [1,2,3,6,8]
y_train = [4.8,8.5,10.4,21.0,25.3]

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter('./tensorflow',sess.graph)

for i in range(10000):
    summary,_ = sess.run([merged,train], {x:x_train,y:y_train})

    writer.add_summary(summary,i)


curr_W,curr_b,curr_loss = sess.run([W,b,loss],{x:x_train,y:y_train})

print("After train W: %s b %s loss %s" % (curr_W,curr_b,curr_loss))