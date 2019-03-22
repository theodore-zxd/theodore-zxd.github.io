---
layout:     post
title:      "深度学习上手"
subtitle:   " \"搭建DnCNN网络进行图像去噪\""
date:       2019-03-20
author:     "Xudong"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - CNN
    - 深度学习
    - 图像去噪
---

## 前言 
根据本人多年来从事算法研究的经验，学习一个或者一类算法最快速的方法就是**实践**。这种“先动手后动脑”的学习模式，往往能带来高效的学习效率。本文就从搭建一种典型的深度学习网络入手，让大家快速入门机器学习算法研究领域。

图像处理是机器学习算法应用最为广泛的领域之一，而**图像去噪**任务是其中较为基础且重要的一种应用，如下图所示。**卷积神经网络**（**CNN**, 深度学习方法的一种）拥有很高的去噪性能，这里我们就搭建一种称为[DnCNN](https://github.com/cszn/DnCNN) [1]的去噪神经网络。读者可以借此机会体验和理解CNN，图像去噪和深度学习等方法机理。
以下是相对路径
![](/img/in-post/dncnn/fig1.png)

## 硬件和系统要求
1. 由于此网络训练和应用过程都会使用到显卡，确保自己电脑拥有块不错性能的独立显卡。另外显卡品牌只能是**NVIDIA**，经过测试性能优于NVIDIA GeForce MX150的显卡即可。
2. **WIN10**系统（本文挑选了一种可以在较为常用的windows系统下运行的网络作为入门，但是大部分深度学习算法都会选择在LINUX系统下搭建，本博客后续的文章也将会以LINUX系统中的算法为主）

## 开始搭建环境

环境搭建时请注意各个软件的版本号，版本对不上的话有时会遇到难以解释的bug。配置环境对于很多算法来说是比较麻烦又不可绕过的一坎。配置过程中我们应该时刻保持警惕，弄错一个步骤就会导致功亏一篑，最后再次重装系统，重头再来。

### 1. 安装CUDA9.1

CUDA是NVIDIA公司开发的显卡驱动工具包，凡是使用GPU并行加速的程序都依赖这个驱动，深度学习也不例外。安装包下载链接：https://developer.nvidia.com/cuda-91-download-archive?target_os=Windows&target_arch=x86_64

参考图中的选项，再点击下载安装包，然后安装。

![](/img/in-post/dncnn/fig2.png)

### 2. 安装cuDNN

cuDNN是用于深度神经网络的GPU加速库，它可以被看做是一个依附于CUDA上的一个额外的库。一般来说，这两者结合才能满足深度学习网络的驱动需求。下载链接：https://developer.nvidia.com/rdp/cudnn-archive，注意要选择与CUDA配对的版本，如下图所示。

![](/img/in-post/dncnn/fig3.png)

cuDNN的安装是以打补丁的方式，主要是3步：

（1）复制CUDA安装目录下<installpath>\cuda\bin\cudnn64_7.dll 文件至 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin文件夹下

（2）复制CUDA安装目录下<installpath>\cuda\ include\cudnn.h 文件至 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include文件夹下

（3）复制CUDA安装目录下<installpath>\cuda\lib\x64\cudnn.lib文件至C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64文件夹下

（4）复制完成后，原先完整的cuDNN安装包存下来不要删除，后面步骤还需要使用。

### 3. 安装VISUAL STUDIO 2015

VS就不用介绍了，可以说是windows下宇宙第一编译器。这里我们使用2015版，community版本就可以了，无需破解。

下载地址：https://visualstudio.microsoft.com/zh-hans/vs/older-downloads/?rr=https%3A%2F%2Fwww.baidu.com%2Flink%3Furl%3D2bJ7RUd1X4L6JArIAU13JVQE-FnrHi5jiu-uzEPUCljNXw5rp37OH-HtIzKoBs5AJ-mOSQRLfz9rOxjWx_Soha%26wd%3D%26eqid%3Da5636d210001544a000000045c923f05

安装完VS还需要安装配套的C++编译器。打开刚才安装的VS2015，新建项目，安装适用于windows桌面的C++工具，如下图所示。

![](/img/in-post/dncnn/fig4.png)

### 4. 安装MATLAB 2017a

注意MATLAB的版本号哦，2016版的我也曾经试过，可行，但是其他版本就不敢保证了。最好是按照本教程选择2017a的MATLAB。由于这一步的安装涉及到破解，因此下载链接可能随时会失效。这里就请读者自行查找靠谱的下载链接了，并且自己破解了。

### 5. MatConvNet安装

敲黑板！注意这里是环境搭建的最重要的一步，也是比较复杂的一步，务必打起精神。

MatConvNet是MATLAB的一个官方定制的工具盒，用来处理卷积神经网络的搭建和使用。它的官方网址是：http://www.vlfeat.org/matconvnet/，上面找到下载链接（版本号：matconvnet-1.0-beta25），下载压缩包。具体安装步骤如下：

（1）**解压**：将压缩包中的MatConvNet解压到某个目录下（解压在C盘的某些文件夹下会有权限问题，建议解压在其他盘，或者桌面上，教程上我们就示范解压在桌面上）。将MATLAB的工作路径置于所解压的文件夹下的matlab文件夹中，如下图。

![](/img/in-post/dncnn/fig5.png)

（2）**设置**：由于MatConNet有部分涉及C++语言编写的代码，因此需要配置MATLAB兼容C++的交叉编译，输入以下命令：

> ```
> > mex -setup C++
> ```
>
> 运行结果如下图所示。

![](/img/in-post/dncnn/fig6.png)

（3）**复制**：之前提到的存下来的cuDNN完整安装包，这里要派上用处啦，因为MatConvNet的安装需要依赖cuDNN。将整个cuDNN文件夹放置在MATLAB安装目录下，如下图。

![](/img/in-post/dncnn/fig7.png)

（4）**编译**：最后输入以下命令完成MatConvNet的编译。

```matlab
> vl_compilenn('enableGpu', true, ...
               'cudaRoot', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', 'C:\Program Files\MATLAB\cudnn') ;
```

上面cudaRoot是安装的CUDA的文件目录，cudnnRoot是上一步复制的文件目录，可以根据自己安装情况进行修改。然后出现以下提示就是成功完成安装了。

![](/img/in-post/dncnn/fig8.png)

**注意：之后每次打开MATLAB都需要进入.../matconvnet/matlab的目录下，运行上述编译代码，然后才能使用整个CNN工具箱进行深度学习研究。**

（5）**验证**：在MATLAB命令行输入以下命令，测试之前几部的安装是否正确，这时你可以听听显卡呻吟的声音:

```matlab
> vl_testnn('gpu', true)
```

测试时间比较久，耐性等待之后，大家就可以开始深度学习之旅了！兴奋地搓着小手.gif

## 6. 运行DnCNN

首先从github上下载源码https://github.com/cszn/DnCNN，解压。将MATLAB的工作目录设为../DnCNN-master/TrainingCodes/DnCNN_TrainingCodes_DagNN_v1.1，然后运行测试脚本Demo_Test_DnCNN_DAG.m,接下来大家就能看到网络对12张测试图片的去澡效果啦。

![](/img/in-post/dncnn/fig9.png)

左图为加噪声图像，中间为原始图像，右图为去噪后图像，可以看到CNN还是比较给力的，这样的去噪效果传统方法是达不到的。

了解深度学习的朋友都知道，重要的还是训练过程，那么这个网络怎么训练它呢？

（1）我们将MATLAB工作目录至于.../DnCNN-master/TrainingCodes/DnCNN_TrainingCodes_v1.0/data，运行GenerateData_model_64_25_Res_Bnorm_Adam.m脚本以生成批量训练的数据对。这个脚本是将当前目录下Train400文件夹下的图像进行分批，加噪声的方式组成无噪声和有噪声的两种图像数据对，用来后续训练神经网络。

（2）完成以后，进入上一级目录，运行Demo_Train_model_64_25_Res_Bnorm_Adam.m脚本，这时候就开始训练神经网络了。这里需要注意，有些显卡显存不够的话，训练一段时间会报错，显示“显存不够”。这时候就需要一些高端操作来解决这个问题了，比如更改神经网络的层数等等。考虑到本文篇幅问题，这些操作就不在这边细说了。测试下来GTX 1070以上的显卡还是能够胜任这个工作的。训练时间不长也不短，几个小时还是要的。

（3）完成训练以后，在运行当前目录下的Demo_Test_model_64_25_Res_Bnorm_Adam.m脚本就能获得对新训练模型的测试结果了。

测试过程中我们可以发现，还有很多脚本，有的却不能运行。其中的原因交给读者慢慢寻找原因啦，找原因的过程也是一个高效学习的过程。

## 7. 总结

以上便是，在windows下搭建DnCNN的全部详细流程。希望借此机会能够让读者入门深度学习，感受到算法的魅力。之后本博客还会更新其他一些算法的搭建流程，后续还会讲解算法原理，敬请期待。


-------------
参考文献：
[1] Zhang K , Zuo W , Chen Y , et al. Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising[J]. IEEE TRANSACTIONS ON IMAGE PROCESSING.