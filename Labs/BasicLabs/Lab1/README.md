# Lab 1 - 框架及工具入门示例

## 实验目的

1. 了解深度学习框架及工作流程（Deep Learning Workload）
2. 了解在不同硬件和批大小（batch_size）条件下，张量运算产生的开销


## 实验环境

* PyTorch==1.5.0

* TensorFlow>=1.15.0

* 【可选环境】 单机Nvidia GPU with CUDA 10.0


## 实验原理

通过在深度学习框架上调试和运行样例程序，观察不同配置下的运行结果，了解深度学习系统的工作流程。

## 实验内容

### 实验流程图

![](/imgs/Lab1-flow.png "Lab1 flow chat")

### 具体步骤

1.	按装依赖包。PyTorch==1.5, TensorFlow>=1.15.0

2.	下载并运行PyTorch仓库中提供的MNIST样例程序。

3.	修改样例代码，保存网络信息，并使用TensorBoard画出神经网络数据流图。

4.	继续修改样例代码，记录并保存训练时正确率和损失值，使用TensorBoard画出损失和正确率趋势图。

5.	添加神经网络分析功能（profiler），并截取使用率前十名的操作。

6.	更改批次大小为1，16，64，再执行分析程序，并比较结果。

7.	【可选实验】改变硬件配置（e.g.: 使用/ 不使用GPU），重新执行分析程序，并比较结果。


## 实验报告

### 实验环境

||||
|--------|--------------|--------------------------|
|硬件环境|CPU（vCPU数目）|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
||GPU(型号，数目)||
|软件环境|OS版本||
||深度学习框架<br>python包名称及版本||
||CUDA版本||
||||

### 实验结果

1. 模型可视化结果截图
   
|||
|---------------|---------------------------|
|<br/>&nbsp;<br/>神经网络数据流图<br/>&nbsp;<br/>&nbsp;|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
|<br/>&nbsp;<br/>损失和正确率趋势图<br/>&nbsp;<br/>&nbsp;||
|<br/>&nbsp;<br/>网络分析，使用率前十名的操作<br/>&nbsp;<br/>&nbsp;||
||||


2. 网络分析，不同批大小结果比较

下面为只考虑推理阶段的结果

|||
|------|--------------|
|批大小 &nbsp;| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 结果比较 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
|<br/>&nbsp;<br/>1<br/>&nbsp;<br/>&nbsp;|<br/>&nbsp;<br/>batchsize为1时，可以看到最耗时的操作conv2d、convolution、_convolution等耗时约在3s，在推理阶段这些耗时的算子的执行次数为20000次，而验证数据集为10k张图片，因此计算图中有两个卷积层；目前使用`profile_memory`的参数设置会出错，还无法看内存信息<br/>&nbsp;<br/>&nbsp;|
|<br/>&nbsp;<br/>16<br/>&nbsp;<br/>&nbsp;|<br/>&nbsp;<br/>batchsize为16时，可以看到相比与batchsize为1时，最耗时操作的总执行时间有了大幅下降，从3s左右，下降到0.4~0.5s。每个操作的平均执行时间从150us上升到350us，平均执行时间变为2.3倍，而计算量却提高了16倍；此处仍无法对内存进行分析；<br/>&nbsp;<br/>&nbsp;|
|<br/>&nbsp;<br/>64<br/>&nbsp;<br/>&nbsp;|<br/>&nbsp;<br/>在batchsize为64时，对于最耗时操作提升有限；总执行时间继续下降到0.3s左右，平均执行每步从350us升高到960us，增为原来的2.7倍，计算量只增为原来的4倍。<br/>&nbsp;<br/>&nbsp;|
|||

## 参考代码

1.	MNIST样例程序：

    代码位置：Lab1/mnist_basic.py

    运行命令：`python mnist_basic.py`

2.	可视化模型结构、正确率、损失值

    代码位置：Lab1/mnist_tensorboard.py

    运行命令：`python mnist_tensorboard.py`

3.	网络性能分析

    代码位置：Lab1/mnist_profiler.py

## 参考资料

* 样例代码：[PyTorch-MNIST Code](https://github.com/pytorch/examples/blob/master/mnist/main.py)
* 模型可视化：
  * [PyTorch Tensorboard Tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) 
  * [PyTorch TensorBoard Doc](https://pytorch.org/docs/stable/tensorboard.html)
  * [pytorch-tensorboard-tutorial-for-a-beginner](https://medium.com/@rktkek456/pytorch-tensorboard-tutorial-for-a-beginner-b037ee66574a)
* Profiler：[how-to-profiling-layer-by-layer-in-pytroch](https://stackoverflow.com/questions/53736966/how-to-profiling-layer-by-layer-in-pytroch)


## 原始数据
### 推理阶段的profiler数据
#### test batch size = 1
---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  
Name                         Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     Number of Calls  
---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  
conv2d                       1.27%            99.194ms         40.59%           3.172s           158.578us        20000            
convolution                  1.01%            79.083ms         39.32%           3.072s           153.618us        20000            
_convolution                 5.17%            404.232ms        38.31%           2.993s           149.664us        20000            
cudnn_convolution            21.74%           1.699s           21.74%           1.699s           84.936us         20000            
addmm                        17.99%           1.406s           17.99%           1.406s           70.290us         20000            
relu                         11.01%           860.435ms        11.01%           860.435ms        28.681us         30000            
add                          8.11%            633.551ms        8.11%            633.551ms        31.678us         20000            
pin_memory                   4.90%            382.602ms        8.10%            633.297ms        31.665us         20000            
to                           5.32%            415.549ms        6.59%            514.740ms        17.158us         30000            
max_pool2d                   0.76%            59.394ms         4.91%            383.995ms        38.399us         10000            
---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  
上面为不按输入形状进行分类
#### test batch size = 16
---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  
Name                         Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     Number of Calls  
---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  
conv2d                       0.89%            7.117ms          56.55%           453.209ms        362.567us        1250             
convolution                  0.76%            6.067ms          55.67%           446.091ms        356.873us        1250             
_convolution                 3.88%            31.126ms         54.91%           440.024ms        352.019us        1250             
cudnn_convolution            43.08%           345.260ms        43.08%           345.260ms        276.208us        1250             
addmm                        11.68%           93.602ms         11.68%           93.602ms         74.882us         1250             
pin_memory                   6.25%            50.076ms         8.78%            70.326ms         56.261us         1250             
relu                         7.08%            56.754ms         7.08%            56.754ms         30.269us         1875             
add                          5.64%            45.181ms         5.64%            45.181ms         36.145us         1250             
to                           4.41%            35.357ms         5.46%            43.745ms         23.331us         1875             
max_pool2d                   0.52%            4.186ms          3.41%            27.328ms         43.725us         625              
---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  
#### test batch size = 64
---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  
Name                         Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     Number of Calls  
---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  
conv2d                       0.48%            1.956ms          73.87%           303.437ms        966.359us        314              
convolution                  0.41%            1.673ms          73.39%           301.481ms        960.130us        314              
_convolution                 2.12%            8.703ms          72.99%           299.808ms        954.803us        314              
cudnn_convolution            66.36%           272.610ms        66.36%           272.610ms        868.184us        314              
pin_memory                   4.86%            19.958ms         6.53%            26.806ms         85.368us         314              
addmm                        6.51%            26.747ms         6.51%            26.747ms         85.181us         314              
relu                         3.92%            16.112ms         3.92%            16.112ms         34.209us         471              
to                           2.85%            11.696ms         3.39%            13.923ms         29.561us         471              
add                          3.22%            13.230ms         3.22%            13.230ms         42.134us         314              
max_pool2d                   0.29%            1.180ms          1.91%            7.829ms          49.868us         157              
---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  