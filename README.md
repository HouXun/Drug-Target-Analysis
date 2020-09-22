Drug-Target-Analysis
====

基于网络的药物-靶标相互作用分析<br>
作者：侯逊<br>
----

# 项目背景
药物研究需要耗费企业大量的时间和金钱成本，也伴随着潜在的失败风险。近年来，药物重定位成为研究热点，即根据已有的实验相关数据，分析现有药物潜在的新靶标，从而发掘药物的新功能。本文采用基于神经网络的药物-靶标相互作用分析模型，根据已有的药物-靶标相互反应数据，预测现有药物潜在的新靶标。本课题主要完成以下的研究工作：<br>

(1) 由于输入数据集存在类别不平衡问题（大量的'0'标签和少量的'1'标签），需要对输入数据集加入带偏移权重。<br>

(2) 采用RandomRWR算法分别提取药物、靶标数据集的特征。<br>

(3) 搭建神经网络模型，网络最顶层采用sigmoid激活函数，输出某药物-靶标对的相互反应概率值（0~1）。<br>

(4) 根据模型的一系列指标（召回率、均方差、ROC、AUC）评估模型效果，并进行模型调优。<br>
## 系统框架
![系统框架如图所示](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/structure.png)

## 数据获取
社交平台：[Tumblr](https://tumblr.zendesk.com/hc/zh-cn)<br>

## 预测结果
![test1](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/test1.png)<br>
![test2](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/test2.png)<br>
![test3](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/test3.png)<br>
![test4](https://github.com/HouXun/Image-Emotional-Analysis/raw/master/pics/test4.png)<br>




