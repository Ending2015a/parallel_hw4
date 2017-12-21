import tensorflow as tf

def hang():
    try:
        while True:
            import time
            time.sleep(5)
    except KeyboardInterrupt:
        try:
            import time
            time.sleep(3)
        except KeyboardInterrupt:
            exit()

def getGPU():
    import tensorflow as tf
    sess = tf.Session()
    hang()
    sess.close()

while True:
    print('gpu hanging')
    getGPU()
    print('gpu release')
    hang()
