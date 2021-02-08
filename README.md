# DAFAR-Prototype

[TOC]

## DAFAR: Detecting Adversaries by Feedback-Autoencoder Reconstruction --- Prototyping System

Deep learning has shown impressive performance on challenging perceptual tasks. However, researchers found deep neural networks vulnerable to adversarial examples. Since then, many methods are proposed to defend against or detect adversarial examples, but they are either attack-dependent or shown to be ineffective with new attacks.

We propose DAFAR, a feedback framework that allows deep learning models to detect adversarial examples in high accuracy and universality. DAFAR has a relatively simple structure, which contains a target network, a plug-in feedback network and an autoencoder-based detector. The key idea is to capture the high-level features extracted by the target network, and then reconstruct the input using the feedback network. These two parts constitute a feedback autoencoder. It transforms the imperceptible-perturbation attack on the target network directly into obvious reconstruction-error attack on the feedback autoencoder. Finally the detector gives an anomaly score and determines whether the input is adversarial according to the reconstruction errors. Experiments are conducted on MNIST and CIFAR-10 data-sets. Experimental results show that DAFAR is effective against popular and arguably most advanced attacks without losing performance on legitimate samples, with high accuracy and universality across attack methods and parameters.

## 使用与测试方法

###  项目文件夹架构

- utils: 工具与工作文件夹
  - dataset: 存放已准备好的 MNIST 对抗样本数据集
    - cw: CW 攻击的测试集对抗样本数据集
    - fgsm: FGSM 攻击的测试集对抗样本数据集，文件名 “fgsmi.npy” 中的 “i” 指攻击强度 epsilon 是 0.05 的多少倍
    -  pgd: PGD 攻击的测试集对抗样本数据集
  - mnist: 已下载好的 MNIST 数据集
  - model: 存放已训练好的模型参数
    - DETECTOR: 存放对抗样本检测器 detector $C(\cdot)$ 的模型参数（MSTDtcAnomL2.pth）
    - MNIST: 存放目标分类器（Tclassifier.pth）与反馈重构网络（Decoder.pth）的模型参数
  - examples: 示例中使用的图片
    - 1.jpg: 1 的正常样本
    - 1a.jpg: 1 的对抗样本
    - 6.jpg: 6 的正常样本
    - 6a.jpg: 6a的对抗样本
  - Architectures.py: 保存了对抗样本检测器、目标分类器与反馈重构网络的模型结构
  - ATTACKS.py: 保存了对抗样本生成函数，可生成FGSM、CW、PGD的对抗样本
  - MSTDtcAnom.py: 训练对抗样本检测器
  - TargetTrain.py: 训练目标分类器与反馈重构网络
  - mydataloader.py: 数据集类
  - Prototype-configure.py: 计算 Anomaly Score 阈值
  - Prototype-runtime.py: 判断一个样本是否为对抗样本
  - Prototype-test.py: 测试该原型系统对于对抗样本的检出率与对正常样本的假阳性率
- README.md
- DAFAR原型系统开发计划.pptx: 介绍了该原型系统的结构与功能

### 使用方法

1. 首先将目标网络、反馈重构网络与对抗样本检测器的结构保存在 Architectures.py 中。其中目标网络、反馈重构网络设计为一体，命名为 `MSTreAE`，对抗样本检测器结构命名为 `MSTDtcAnom`。
2. 将用正常样本训练好的各模型参数存放在 model 文件夹中，按已有模型参数命名。
3. **计算 Anomaly Score 阈值**：在命令行终端输入 `python3 Prototype-configure.py`，等待片刻程序便会在终端打印出阈值的数值。
4. **判断一个样本是否为对抗样本**：在命令行终端输入 `python3 Prototype-runtime.py -i INPUT -t THRESHOLD`，其中 `INPUT` 是输入的图片路径，`THRESHOLD` 是刚才计算出的阈值。等待片刻程序便会在终端打印出判断结果：对于正常样本，输出正确分类标签；对于对抗样本，输出警告信息。
5. **测试原型系统准确度**：在命令行终端输入 `python3 Prototype-runtime.py -y TYPE -i INPUT -t THRESHOLD`，其中 `TYPE` 是待测试数据集类型，为 `adversarial` 或 `normal`；`INPUT` 为待测试数据集路径（对抗样本数据集已经提前生成好，可以直接使用）；`THRESHOLD` 是刚才计算出的阈值。等待片刻程序便会在终端打印出结果：对于对抗样本的检出率或对正常样本的假阳性率。

