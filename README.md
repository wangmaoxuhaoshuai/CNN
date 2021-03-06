# CNN
There are some codes about CNN.

## VGG
![Image text](https://github.com/wangmaoxuhaoshuai/CNN/blob/master/images/VGG.png)

1.VGG 的细节之 3x3 卷积核

>VGG 和 AlexNet 最大的不同就是 VGG 用大量的 3x3 卷积核替换了 AlexNet 的卷积核。
3x3 卷积核是能够感受到上下、左右、重点的最小的感受野尺寸。并且，2 个 3x3 的卷积核叠加，它们的感受野等同于 1 个 5x5 的卷积核，3 个叠加后，它们的感受野等同于 1 个 7x7 的效果
既然，感受野的大小是一样的，那么用 3x3 有什么好处呢?
答案有 2，一是参数更少，二是层数加深了。
现在解释参数变少的问题。
假设现在有 3 层 3x3 卷积核堆叠的卷积层，卷积核的通道是 C 个，那么它的参数总数是 3x(3Cx3C) = 27C^2
同样和它感受野大小一样的一个卷积层，卷积核是 7x7 的尺寸，通道也是 C 个，那么它的参数总数就是 49C^2
通过计算很容易得出结论，3x3 卷积方案的参数数量比 7x7 方案少了 81% 多，并且它的层级还加深了。

2.VGG 的细节之 1x1 卷积核
>堆叠后的 3x3 卷积层可以对比之前的常规网络的基础上，减少参数数量，而加深网络。
但是，如果我们还需要加深网络，怎么办呢？
堆叠更多的的卷积层，但有 2 个选择。
选择 1：继续堆叠 3x3 的卷积层，比如连续堆叠 4 层或者以上。
选择 2：在 3x3 的卷积层后面堆叠一层 1x1 的卷积层。
1x1 卷积核的好处是不改变感受野的情况下，进行升维和降维，同时也加深了网络的深度。

3.VGG 其它细节汇总
>大家一般会听说 VGG-16 和 VGG-19 这两个网络，其中 VGG-16 更受欢迎。
16 和 19 对应的是网络中包含权重的层级数，如卷积层和全连接层，大家可以仔细观察文章前面贴的配置图信息。
所有的 VGG 网络中，卷积核的 stride 是 1，padding 是 1.
max-pooling 的滑动窗口大小是 2x2 ，stride 也是 2.

原文：https://blog.csdn.net/briblue/article/details/83792394 

