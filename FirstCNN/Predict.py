import tensorflow as tf
from PIL import Image, ImageFilter
from CNN import NetworkFramework
import os
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def predictInt(imgValue):
    tf.reset_default_graph()
    x, y, y_conv, var_list, keep_prob = NetworkFramework()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(var_list)

    sess = tf.InteractiveSession()
    sess.run(init)
    saver.restore(sess, 'Model/model.ckpt')

    prediction = tf.argmax(y_conv, 1)
    return prediction.eval(feed_dict={x: [imgValue], keep_prob: 0.5}, session=sess)

def imageprepare(argv):
    #Image.convert() 转换图像 'L'表示灰度图像
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), 255)

    if width > height:
        #round(number,digits)对number按digits的位数进行四舍五入
        #这里的20/除以面积是为了把不规则图片化为20xNone的图片
        nheight = int(round((20.0/width*height), 0))
        if nheight == 0:
            nheight = 1
        #Image.ANTIALIAS 抗锯齿
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))
        #paste(img,loc) img为要复制的图像，loc为坐标
        newImage.paste(img, (4, wtop))
    else:
        nwidth = int(round((20.0 / width * height), 0))
        if nwidth == 0:
            nwidth = 1
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round((28 - width) / 2, 0))
        newImage.paste(img, (wleft, 4))

    tv = list(newImage.getdata())
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva

# use our own image for test
def main():
    for i in range(10):
        imvalue = imageprepare('Test_data/%d.jpg'%(int(i)))
        predint = predictInt(imvalue)
        print(predint[0])

# use MNIST for test
def MNIST_test():
    mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)

    x, y, y_conv, var_list, keep_prob = NetworkFramework()

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(var_list)

    sess = tf.InteractiveSession()
    sess.run(init)
    saver.restore(sess, 'Model/model.ckpt')

    for i in range(100000):
        batch = mnist.test.next_batch(60)
        if i % 1000 == 0:
            print('test accuracy %g' % sess.run(accuracy, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5}))

if __name__ == '__main__':
    # main()
    MNIST_test()

