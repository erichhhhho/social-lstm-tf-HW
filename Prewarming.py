import tensorflow as tf

from tensorflow.python.ops import init_ops
from tensorflow.python.client import timeline
import logging, time
import numpy as np

def test():
    n = 5000

    with tf.device('/gpu:0'):
        A1 = tf.Variable(tf.ones_initializer(shape=[n, n]), name='A1')
        B1 = A1
        for l in range(10):
            B1 = tf.matmul(A1, B1, name="chain1")

    with tf.device('/gpu:1'):
        A2 = tf.Variable(tf.ones_initializer(shape=[n, n]), name='A2')
        B2 = A2
        for l in range(10):
            B2 = tf.matmul(A2, B2, name="chain2")
        C = tf.matmul(B1, B2)

    run_metadata = tf.RunMetadata()
    start = time.time()
    logging.info('started')
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=True))
    sess.run(tf.initialize_all_variables())
    # do warm-run
    sess.run([C.op],
             options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
             run_metadata=run_metadata)
    run_metadata = tf.RunMetadata()
    sess.run([C.op],
             options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
             run_metadata=run_metadata)
    logging.info('writing trace')
    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
    trace_file = open('timeline.ctf.json', 'w')
    trace_file.write(trace.generate_chrome_trace_format(show_memory=True))
    logging.info('trace written')
    end = time.time()
    logging.info('computed')
    logging.info(end - start)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    test()