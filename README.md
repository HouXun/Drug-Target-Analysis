Drug-Target-Analysis
====

基于网络的药物-靶标相互作用分析<br>
作者：侯逊<br>
----

# 项目背景
药物研究需要耗费企业大量的时间和金钱成本，也伴随着潜在的失败风险。近年来，药物重定位成为研究热点，即根据已有的实验相关数据，分析现有药物潜在的新靶标，从而发掘药物的新功能。本文采用基于神经网络的药物-靶标相互作用分析模型，根据已有的药物-靶标相互反应数据，预测现有药物潜在的新靶标。本课题主要完成以下的研究工作：<br>

(1) 采用RandomRWR算法分别提取药物、靶标数据集的特征。<br>

(2) 由于输入数据集存在类别不平衡问题（大量的'0'标签和少量的'1'标签），需要对输入数据集加入带偏移权重。<br>

(3) 搭建神经网络模型，网络最顶层采用sigmoid激活函数，输出某药物-靶标对的相互反应概率值（0~1）。<br>

(4) 根据模型的一系列指标（召回率、均方差、ROC、AUC）评估模型效果，并进行模型调优。<br>
## 系统框架
![系统框架如图所示](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/structure.png)

## 数据获取
社交平台：[Tumblr](https://tumblr.zendesk.com/hc/zh-cn)<br>
API: [PyTumblr](https://pypi.org/project/PyTumblr/)<br>

## 图像预处理
* 人工排除无关图像和不一致标签图像。无关图像包括广告、无效图像等。不一致标签图像反映出用户发表的图像和文本所表达的情感存在不一致现象，通常情况下文本所表达的情感是用户的实时发帖情感，而图像往往不能准确的表达出该情感。如图所示，该图展现了一位正在哭泣的女人，然而文本部分表明这位女人见到了奥巴马喜极而泣；下图描述的是一位微笑的男孩，但根据文本，事实上是这位小男孩不幸去世。对于这种不一致标签的图像，也需要进行人工检查筛选。<br>
![不一致标签图像](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/noise1.png)<br>

* 特定算法移除特征不明显图像。特征不明显图像的特点是图像颜色过于单一、在纯色背景下附有大量的文字。如图所示，虽然从图片中的文字能看出用户发帖时的情感是积极、乐观的，但从图像本身分析，该图像并不具备表达积极情感的特征。<br>
![特征不明显图像](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/noise2.jpg)<br>
![特征不明显图像筛选算法](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/algorithm1.png)<br>
## 图像特征提取
### 传统图像特征
* HSV颜色直方图。HSV颜色空间由色调、饱和度和亮度组成。色调H表示颜色种类，饱和度表示颜色深浅，亮度V表示颜色明暗程度。HSV颜色空间是最符合人眼视觉特性的颜色模型，在图像分类、检索中，应用这种颜色模型会更适合用户的视觉判断。<br>
以全局颜色直方图作为图像颜色特征，仅包含了图像的整体颜色分布，缺少了图像的空间信息。因此，为反映出颜色的空间分布关系，本系统引入了HSV局部特征直方图，即将整幅图片四等分块，分别计算每一块的HSV颜色直方图，最后将四个直方图横向拼接成局部颜色直方图，作为该图像的颜色特征。<br>
![hsv](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/hsv.png)<br>
* 灰度共生矩阵。灰度共生矩阵的定义为：任取图像中的一组点对(x，y)和(x+a，y+b)，设该点对的灰度值对为(p，q)，若灰度值有n种取值，则(p，q)的组合共有 n2种。对于整个图像像素矩阵，统计出每一种(p，q)值出现的次数并建立概率矩阵，这样的矩阵称为灰度共生矩阵。当a=1，b=0时，以0度方向统计像素对；当a=0，b=1时，以90度方向统计像素对；当a=1，b=1时，以45度方向统计像素对。<br>
本系统定义最大灰度级数为16，并分别计算出图像0度、45度、90度扫描的灰度共生矩阵，并计算每个矩阵的角二阶矩、熵、对比度、均匀度这四个参数，三组参数拼接成一组12维的特征向量，作为该图像的纹理特征。<br>

* HOG和LBP特征。HOG（方向梯度直方图）计算统计图像的局部梯度方向直方图作为特征。LBP（局部二值模式）提取图像的局部纹理特征，其具有旋转不变性和灰度不变性等优点。<br>

### 预训练Inception网络特征
本系统使用预训练的InceptionV3网络提取图像高层次特征。以往的神经网络大多是通过增加网络层数来获得更好的训练效果，但网络深度的增加会带来过拟合、计算开销庞大、梯度消失、梯度爆炸等问题。InceptionV3则通过更改网络宽度、深度、卷积核等提高模型效果，可以更高效地利用计算资源，提取到更多的特征，从而提升训练结果。<br>

实现库：[Keras](https://keras.io/)<br>

### 特征拼接
本模块提出了传统特征与预训练网络特征结合的创新方法。即将四种传统特征和Inception网络提取的高层次特征拼接在一起，作为图像的整体特征。特征拼接后，每个图像对应一个4394维的特征向量。<br>

## 模型训练
本系统本搭建了一个三层神经网络，网络结构如图所示。该网络由两个全连接层和一个Dropout层组成，接收形状为(samples，4394)的numpy向量，返回形状为(samples，15)的numpy向量，代表每个图像对应的15类情感概率分布。模型搭建后需要进行编译、训练，由于本系统处理的是多分类问题，需要用rmsprop优化器和交叉熵损失函数配置模型，在训练过程中记录验证精度。最后反复训练得到最优模型，并保存到本地。<br>
![模型网络结构](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/model.png)<br>

## 数据库设计
### 用户表
![用户表](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/user_table.png)<br>
### 图像表
![图像表](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/img_table.png)<br>
## Django框架
Django是一个基于python的Web框架。Django框架能自动处理用户和控制器的交互部分，其核心是模型(Model)、模板(Template)和视图(Views)，因此Django采用的是MTV模式。<br>
Django的优势在于它能简单、快速实现一个具有前端、后端、数据库的网站，通过Django内置函数可以很容易地实现用户与系统交互、业务逻辑、数据库搭建等模块。同时Django内置许多功能齐全的插件，如验证码等，具有很强的可扩展性。<br>
![Django](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/django.png)<br>

## 预测结果
![test1](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/test1.png)<br>
![test2](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/test2.png)<br>
![test3](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/test3.png)<br>
![test4](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/test4.png)<br>




