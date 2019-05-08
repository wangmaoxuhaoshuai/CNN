import tensorflow as tf
from PIL import Image, ImageFilter
import os
tf.reset_default_graph()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def predictInt(imgValue):

    x = tf.placeholder(tf.float32, [None, 784])

    def weight_var(shape, name):
        return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

    def bias_var(shape, name):
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(0))

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pol_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_var([5, 5, 1, 6], 'W_conv1')
    b_conv1 = bias_var([6], 'b_conv1')
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pol_2x2(h_conv1)

    W_conv2 = weight_var([5, 5, 6, 16], 'W_conv2')
    b_conv2 = bias_var([16], 'b_con2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pol_2x2(h_conv2)

    W_conv3 = weight_var([5, 5, 16, 120], 'W_conv3')
    b_conv3 = bias_var([120], 'b_conv3')
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 120])

    W_fcl1 = weight_var([7 * 7 * 120, 1024], 'W_fcl1')
    b_fcl1 = bias_var([1024], 'b_fcl1')
    h_fcl1 = tf.nn.relu(tf.nn.xw_plus_b(h_pool3_flat, W_fcl1, b_fcl1))
    keep_prob = tf.placeholder(tf.float32)
    h_fcl1_drop = tf.nn.dropout(h_fcl1, keep_prob)

    W_fcl2 = weight_var([1024, 10], 'W_fcl2')
    b_fcl2 = bias_var([10], 'b_fcl2')
    y_conv = tf.nn.softmax(tf.nn.xw_plus_b(h_fcl1_drop, W_fcl2, b_fcl2))

    var_list = [
        W_conv1, b_conv1,
        W_conv2, b_conv2,
        W_conv3, b_conv3,
        W_fcl1, b_fcl1,
        W_fcl2, b_fcl2
    ]

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(var_list)

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, 'Model/model.ckpt')

        prediction = tf.argmax(y_conv, 1)
        result = prediction.eval(feed_dict={x: [imgValue], keep_prob: 1.0}, session=sess)
    sess.close()
    return result

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

def main(argv):
    imvalue = imageprepare(argv)
    predint = predictInt(imvalue)
    print(predint[0])

if __name__ == '__main__':
    main('Test_data/9.jpg')

