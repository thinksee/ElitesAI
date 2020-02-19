## [Pytorch基础](./notebooks/00_pytorch_basic.ipynb)

主要包括创建Tensor的方式、修改Tensor的形状、索引Tensor、广播机制、逐元素操作、归并操作、矩阵操作等。

## [线性回归模型](./notebooks/01_linear_model.ipynb)

手动实现一个简单的线性回归模型包括组件：数据读写器、模型参数、模型定义、损失函数、优化函数等；

利用pytorch实现线性模型采用优化器SGD，损失函数MSE，线性回归使用的都是类似的单层神经网络结构。

对比上，Pytorch整合代码简洁，但是时间上并不如意。

其中损失函数和梯度下降方向一直，若要求解的目标值是最小值，则利用梯度下降的方式，若要求解的目标是最大值时，则利用梯度上升的方式。

## [Softmax and Classifier model](./notebooks/02_softmax_classifier.ipynb)

Softmax把NN模型的输出值变换成值为正且和为1的概率分布，从无法解释的输出转化为基于概率的可解释输出。

[交叉熵损失函数](https://www.cnblogs.com/kyrieng/p/8694705.html)

## [MLP](./notebooks/03_multilayer_perceptron.ipynb)

**激活函数**

- 使用激活函数，可以引入非线性性质，提升整体的网络的表达能力。深度学习最主要的特点就是：[多层和非线性](https://cloud.tencent.com/developer/article/1384762)

[**常用激活函数对比**](https://juejin.im/entry/58a1576e2f301e006952ded1)

- Sigmoid主要有两个缺点：**函数饱和度使得梯度消失**，主要是由于sigmoid神经元在值为0或1的时候接近饱和，使得这些区域的梯度几乎为0，因此在反向传播的时候，这个局部梯度会与整个代价函数关于该单元输出的梯度相乘，结果接近为0；**Sigmoid函数不是关于原点中心对称的**，这个特性会导致后面网络层的输入也不是零中心的，进而影响梯度下降的运作。
- tanh 函数同样存在饱和问题，但它的输出是零中心的，因此实际中 tanh 比 sigmoid 更受欢迎。tanh 函数实际上是一个放大的 sigmoid 函数，数学关系为：[$$1-2Sigmoid(x)=tanh(\frac{x}{2})$$](https://blog.csdn.net/yaoyaoyao2/article/details/73848983)
- 相较于 sigmoid 和 tanh 函数，ReLU 对于 SGD 的收敛有巨大的加速作用（[Alex Krizhevsky](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) 指出有 6 倍之多）。有人认为这是由它的线性、非饱和的公式导致的。相比于 sigmoid/tanh，ReLU 只需要一个阈值就可以得到激活值，而不用去算一大堆复杂的（指数）运算。ReLU 的缺点是，它在训练时比较脆弱并且**可能“死掉”**。举例来说：一个非常大的梯度经过一个 ReLU 神经元，更新过参数之后，这个神经元再也不会对任何数据有激活现象了。如果这种情况发生，那么从此所有流过这个神经元的梯度将都变成 0。也就是说，这个ReLU 单元在训练中将不可逆转的死亡，导致了数据多样化的丢失。实际中，如果学习率设置得太高，可能会发现网络中 40% 的神经元都会死掉（在整个训练集中这些神经元都不会被激活）。合理设置学习率，会降低这种情况的发生概率。
- Leaky ReLU 是为解决“ ReLU 死亡”问题的尝试。有些研究者的论文指出这个激活函数表现很不错，但是其效果并不是很稳定。
- Maxout 是对 ReLU 和 Leaky ReLU 的一般化归纳，它的函数公式是（二维时）：。ReLU 和 Leaky ReLU 都是这个公式的特殊情况（比如 ReLU 就是当 时）。这样 Maxout 神经元就拥有 ReLU 单元的所有优点（线性和不饱和），而没有它的缺点（死亡的ReLU单元）。然而和 ReLU 对比，它每个神经元的参数数量增加了一倍，这就导致整体参数的数量激增。
- 通常来说，很少会把各种激活函数串起来在一个网络中使用的。如果使用 ReLU，那么一定要小心设置 learning rate，而且要注意不要让你的网络出现很多 “dead” 神经元，如果这个问题不好解决，那么可以试试 Leaky ReLU、PReLU 或者 Maxout。最好不要用 sigmoid，可以试试 tanh，不过可以预期它的效果会比不上 ReLU 和 Maxout。

## [过拟合和欠拟合](./notebooks/04_over_under_fitting.ipynb)

[为什么正则化会降低过拟合](https://blog.csdn.net/u012535605/article/details/83579214)

## [梯段消失和梯度爆炸](./notebooks/05_gradient_vanishing_explosion.ipynb)

## [卷积神经网络基础](./notebooks/06_cnn_basic.ipynb)

## [LeNet](./notebooks/07_lenet.ipynb)

## [卷积神经网络进阶](./notebooks/09_bn_resnet.ipynb)

包括BN层以及左右和ReNet的基本介绍

## [文本预处理](./notebooks/10_predeal_text.ipynb)

## [语言模型](./notebooks/11_language_model.ipynb)

## [循环神经网络](./notebooks/12_rnn.ipynb)

## [门结构](./notebooks/13_modern_rnn.ipynb)

## [机器翻译及相关技术](./notebooks/14_translation.ipynb)

##  [注意力机制和Seq2Seq](./notebooks/15_attention_seq2seq.ipynb)

## [Transformer](./notebooks/16_transformer.ipynb)

## [凸优化](./notebooks/17_convexity_optimize.ipynb)

## [梯度下降](./notebooks/18_gradient_descent.ipynb)

## [凸优化进阶](./notebooks/19_optimizer_advanced.ipynb)

## [数据增强](./notebooks/20_data_augmentation.ipynb)

## [微调](./notebooks/21_fine_tuning.ipynb)

## [目标检测](./notebooks/22_target_detection.ipynb)

## [word2vec](./notebooks/23_word2vec.ipynb)

## [词嵌入](./notebooks/24_word_embedding.ipynb)

## [问本情感分类](./notebooks/25_text_classifier.ipynb)

## [GAN](./notebooks/26_gan.ipynb)

## [DCGAN](./notebooks/27_dcgan.ipynb)