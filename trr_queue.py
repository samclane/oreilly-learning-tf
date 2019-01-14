import tensorflow as tf

sess = tf.InteractiveSession()
queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])

enqueue_op = queue1.enqueue(["F"])

print(sess.run(queue1.size()))

enqueue_op.run()
print(sess.run(queue1.size()))

enqueue_op = queue1.enqueue(["I"])
enqueue_op.run()
enqueue_op = queue1.enqueue(["F"])
enqueue_op.run()
enqueue_op = queue1.enqueue(["O"])
enqueue_op.run()

print(sess.run(queue1.size()))

# x = queue1.dequeue()
#print(x.eval())
#print(x.eval())
#print(x.eval())
#print(x.eval())

inputs = queue1.dequeue_many(4)
print(inputs.eval())