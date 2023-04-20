# Using CMA-ES to attack the Arbiter PUF and its variants
# 使用CMA-ES算法攻击Arbiter PUF及其变体

## 前言
Arbiter-PUF(APUF)是最早提出的一种基于延时的物理不可克隆电路(PUF)。由于其简单的线性结构，APUF很容易被机器学习的算法攻击。CMA-ES算法是PUF的主流攻击算法之一。该算法可以根据PUF结构自定义适应度(fitness)函数，从而达到较高的预测准确率。本篇文章将使用Python的CMA-ES算法库对APUF及改进的XOR PUF的输出进行预测。

## Arbiter-PUF结构

仲裁器PUF(APUF)电路的结构如下图所示，该电路由多组二选一选择器和一个仲裁器组成。仲裁器一般可采用D触发器或RS锁存器。每两个二选一选择器组成一个传输节点，节点内部上下两个二选一数据选择器必须严格对称摆放，相邻两个节点之间的走线也需严格对称。当一个上升沿信号输入到PUF电路后，信号会分成两路并沿着两条完全对称的路径到达仲裁器。理想状态下，由于上下两条路径完全对称，故两路信号应同时到达仲裁器。然而由于芯片的制造过程中存在随机的工艺误差，故两路信号不可能同时到达仲裁器。根据两路信号到达仲裁器的先后顺序，仲裁器输出0或1。

![传统 APUF 电路结构](https://files.mdnice.com/user/27676/045888cf-220f-46a7-83b1-4cc494a1d4f4.png)

图中所示电路包含128 bit的选择信号(Challenge) X和1 bit的仲裁器输出响应信号(Response) Y。输入信号作为节点内部两个选择器的选择信号，其决定了输入到PUF电路的触发信号的传输路径。从图中可以看到，当选择信号X[0]=0时，输入的上升沿信号平行传至下一级，当X[0]=1时，输入信号交叉传输到下一级。由于每个节点都可以通过选择信号来改变延时路径，因此PUF电路中一共存在 $2^{128}$ 条不同的延时路径。

如下图所示，当改变Challenge导致两路信号到达仲裁器的先后顺序发生反转时，仲裁器便会产生不同输出结果。

![](https://files.mdnice.com/user/27676/5d6dc5c7-13b6-4406-b265-74f7519b4b74.png)

XOR PUF则是使用两个完全相同的APUF，然后用异或门将两个APUF的Response进行异或得到最终的输出响应。



## APUF的数学模型

APUF的数学模型如下图所示（对此部分不感兴趣的读者可以直接跳过）。此模型来自[论文](https://ieeexplore.ieee.org/document/6581579 "Side Channel Modeling Attacks on 65nm Arbiter PUFs Exploiting CMOS Device Noise")。


![](https://files.mdnice.com/user/27676/5164124d-5a77-4d8e-b09a-e8e9b4f95547.png)


![](https://files.mdnice.com/user/27676/03183ff8-de48-4cb2-8ef2-33fdc5b1e79c.png)

图中 $\Delta t_V$ 即两条路径的最终延时差值。而 $\delta t_i^0$ 和 $\delta t_i^1$ 则是第 $i$ 个节点在不同通选择信号时的延时差值。

仲裁器的输出Response为0或1取决于 $\Delta t_V$ 为正或负（对于XOR PUF而言，其输出Response则取决于两个APUF的延时差值的乘积的符号）。 

$\Delta t_V$ 与输入challenge以及每个节点的延时之间的关系如下：
![](https://files.mdnice.com/user/27676/448fe2a3-d9bc-4360-bf82-28e0ee4f255f.png)
其中 $\vec{\tau}$ 是延时向量，需要用CMA-ES算法训练得到。
而 $\vec{\gamma}$ 是特征向量，可由challenge直接计算得到。将challenge换成 $\vec{\gamma}$ 主要为了消除APUF中的非线性，提高模型准确率。 $\vec{\gamma}$ 和 challenge 之间的关系如下：

![](https://files.mdnice.com/user/27676/d9af6c91-6dd2-4169-903d-b4b5412bf53a.png)


# CMA-ES算法预测PUF响应

## 思路

CMA-ES采用协方差矩阵自适应进化策略来找到使目标函数(fitness function)达到最小值的参数向量。如果我们想要用CMA-ES预测PUF的响应，就需要将PUF作为目标函数，该函数的输入是training CRPs，延时向量 $\vec{\tau}$ 即参数向量，输出值是预测错误率。

这样我们就可以用CMA-ES来找到预测错误率最小时的目标函数的延时向量 $\vec{\tau}$ 。然后就可以用数学模型和 $\vec{\tau}$ 预测PUF的输出了。

我们的核心任务就是编写目标函数，该函数能够利用PUF的数学模型和CMA-ES得到的解来计算PUF的Response，然后将其与训练集中正确的response对比，从而计算预测结果的错误率。

接着，我们将错误率反馈给CMA-ES进一步优化延时向量，以获得更低的错误率。该步骤不断重复，直到获得一个符合预期的延时向量为止。


最后，我们用测试集检验最终得到的解的正确性，如果测试集的错误率比训练集高很多，说明出现了过拟合。这时，我们就需要增加训练集和测试集。

具体训练过程如下图：

![](https://files.mdnice.com/user/27676/b6f9dfe6-6ea3-4cb2-966a-df52f0775441.png)


## CMA-ES 类介绍

我们需要使用Python CMA库中的 CMAEvolutionStrategy 类来完成本次实验。该类的使用方法如下：
```python
from cma import CMAEvolutionStrategy

# 创建类
es = CMAEvolutionStrategy(x0, sigma0, opts)

# 开始训练
es.optimize(objective_fct, iterations = 500, args=() )

# 获得训练结果
res = es.result
```
`x0`：初始解；

`sigma0`：进行参数优化时所用的初始标准差；

`opts`：一个字典类型数据，可设置CMA-ES算法的一些参数。例如：最大迭代次数、目标函数最大调用次数等；

`objective_fct`：目标函数，也称作适应度函数或损失函数，该函数的返回值是预测错误率；

`iterations`：最大迭代次数；

`args`：传递给`objective_fct`函数的参数。
**注意：objective_fct 函数的第一个参数必须是CMA-ES所训练得到的解。且该解由optimize函数自动传给objective_fct，不需要通过 args 传递**

训练结果是一个元组类型（tuple），其中每一项内容描述如下：

- 0 ``xbest`` best solution evaluated
- 1 ``fbest`` objective function value of best solution
- 2 ``evals_best`` evaluation count when ``xbest`` was evaluated
- 3 ``evaluations`` evaluations overall done
- 4 ``iterations``
- 5 ``xfavorite`` distribution mean in "phenotype" space, to be
  considered as current best estimate of the optimum
- 6 ``stds`` effective standard deviations, can be used to
  compute a lower bound on the expected coordinate-wise distance
  to the true optimum.

## 步骤

### 1. 安装cma库
``` bash
pip install cma
```
CMA-ES的Python源代码在[这里](https://github.com/CMA-ES/pycma "CMA-ES的Python源代码")
### 2. 编写PUF模型

PUF模型是根据PUF的结构和工作原理建立的软件模型，用于生成训练集和测试集。

```python
def CRP_gen_APUF(num = 12000, stages = 64, xor_num = 1, seed = None):
    '''num    : the needed number of CRPs
       stages : the number of stages in APUF/XOR APUF, i.e. the bit number of challenge
       xor_num: the number of APUFs in XOR APUF. if xor_num=1, it's an APUF
       seed   : Random seed, if it's None, different CRPs will be generated each time.
       '''
    np.random.seed(seed)
    # generate challenges
    chal = np.random.choice([0, 1], [num, stages]).astype(np.int8)
    
    # generate delay parameters
    para_delay = np.random.randn(xor_num, 2, stages)
    
    # create array used to store response
    resp_tot = np.zeros([xor_num, num], dtype = np.int8)
    resp = np.zeros(num, dtype = np.int8)

    for apuf_inx in range(xor_num):
        para = para_delay[apuf_inx]

        # compute the delay difference
        delay_sum = np.zeros(num)
        for i in range(stages):
            para_select = para[chal[:, i], i]
            delay_sum = delay_sum * (1 - 2*chal[:, i]) + para_select

        # generate responses of all the APUF primitives
        resp_tot[apuf_inx, delay_sum >0] = 1
        # XOR operation
        resp = resp ^ resp_tot[apuf_inx]
```
**参数xor_num = 1时, 函数生成APUF的CRP; 而xor_num = 2时, 生成 XOR PUF的CRP; xor_num = n时, 则生成 n-XOR PUF的CRP**

### 2. 编写目标函数
该函数的思路是先通过chal计算出数学模型中的$\vec{\gamma}$, 然后进行矩阵运算$\vec{\gamma} \cdot \vec{\tau}$得到延时差值，并通过判断延时差值的正负来获得response。最后将计算的结果与输入参数 `resp`对比，得到错误率。目标函数的Python代码如下，其中参数`weight` 为CMA-ES得到的解，即数学模型中的向量$\vec{\tau}$。
```python
def fitness_fun(weight, chal, resp, xor_num):
    ''' 
    'chal' and 'resp' are the traing/testing CRP. 
    'weight' are the trained parameters
    '''
    stage_num=chal.shape[1]
    crp_num  =chal.shape[0]

    # according to mathematical model we need stage_num +1 parameters for each APUF primitive
    if weight.shape[0] != (xor_num * (stage_num +1)):
        print('Warning: weight.shape[0] != (xor_num * (stage_num +1)):')
    delay_para = weight.reshape(stage_num+1, xor_num)

    C =1 - 2*chal # switch challenge into [1, -1]

    # transfer the challenge vector into feature vector to avoid nonlinear operation
    C_linear =np.ones((crp_num, stage_num + 1))
    for i in range(stage_num-1, -1, -1):
        C_linear[:,i] = C_linear[:, i+1] * C[:, i]
    
    # calculate the final delay difference
    delay_sum = np.matmul(C_linear, delay_para)  
    
    sum_prod = np.prod(delay_sum, axis = 1)  # equivalent to XOR operation

    resp_predict = np.zeros(crp_num).astype(np.int8)
    resp_predict[sum_prod > 0] = 1

    # calculate the prediction error rate
    err_rate = np.sum(resp ^ resp_predict) / crp_num * 100
    return err_rate
```

### 3. 编写训练程序
fitness函数写好以后，训练程序就比较简单了，具体如下：

**以下代码是测试CMA-ES对APUF的预测准确率，如果测试对 XOR PUF 的预测准确率，只需要将`xor_num`改为2即可**

```python
train_num = 2000
test_num  = 2000
stage_num = 64
xor_num   = 1  # 1-> APUF， 2 -> 2XOR PUF， n -> nXOR PUF

crp_num   = train_num + test_num

############ generate testing set and training set #############
res = CRP_gen_APUF(crp_num, stage_num, xor_num, 10)
print('Uniformity is ', sum(res[1])/crp_num)
chal_train = res[0][:train_num]
resp_train = res[1][:train_num]
chal_test  = res[0][train_num:]
resp_test  = res[1][train_num:]

########## CMA-ES training ################
weight_size = (stage_num + 1) * xor_num
# create CMA-ES class
es = CMAEvolutionStrategy(weight_size * [0], 1)

# start training

# "args" will be sent to "fitness_fun", the "weight" args of "fitness_fun" will be automatically sent in the background by the CMA-ES class
es.optimize(fitness_fun, iterations = 200, verb_disp=10, args = (chal_train, resp_train, xor_num))

res = es.result

########## testing the trained parameters #################
error_rate = fitness_fun(res[0], chal_test, resp_test, xor_num) # res[0] is the trained parameters
print('Prediction Accuracy is: ', 100 - error_rate, '%')
```

  
## 测试结果 Testing results

CMA-ES对于APUF的预测结果如下：

The CMA-ES prediction results on APUF is shown as below:

![APUF training results](https://files.mdnice.com/user/27676/a29a0cf4-9766-4348-b295-44ccbe5c6e8a.png)

可以看到，只需7秒的时间就达到了97%的准确率。

As observed, 97% accuracy is achieved using only 7 seconds.


现在，我们再测试一下CMA-ES对 2-XOR PUF 的预测结果。将`xor_num`改为2，同时训练集增加至4000，测试结果如下：

Now, let's test 2-XOR PUF. We should set `xor_num` to 2 and increase the training set to 4000, the result is shown as below:

![XOR PUF training results](https://files.mdnice.com/user/27676/442f0ffc-59a3-4ec1-9a56-fb422c19022f.png)


即使是3-XOR PUF, 我们用50000的CRP迭代500次也能达到98.9%的准确率:

Even for the 3-XOR PUF, we can still achieve 98.9% accuracy using 50000 training CRPs and iterating 500 times:

![3-XOR PUF training results](https://files.mdnice.com/user/27676/b7089e74-f318-4eb3-a010-80f36d31556f.png)



## 写在最后
自己一直想把CMA-ES预测PUF的程序移植到Python上，看了几天cma库的源码，终于找到了解决办法。如今分享出来，希望能为PUF领域相关科研人员提供帮助，以节省宝贵的科研时间。

这篇技术文档所阐述的CMA-ES攻击方法已被应用在了我们最新的研究工作中，如果此文对您有所帮助，请在您的著作中引用我们的如下研究成果：

The CMA-ES attack method described in this technical document has been applied in our latest research works. If this article is helpful to you, please cite our research work as follows in your work:

```
C. Xu, L. Zhang, M. -K. Law, X. Zhao, P. -I. Mak and R. P. Martins, "Modeling Attack Resistant Strong PUF Exploiting Stagewise Obfuscated Interconnections With Improved Reliability," in IEEE Internet of Things Journal, doi: 10.1109/JIOT.2023.3267657.
```

BIBTEX file:
```latex
@ARTICLE{Xu2022_OIPUF,
  author={Xu, Chongyao and Zhang, Litao and Law, Man-Kay and Zhao, Xiaojin and Mak, Pui-In and Martins, Rui P.},
  journal={IEEE Internet of Things Journal}, 
  title={Modeling Attack Resistant Strong PUF Exploiting Stagewise Obfuscated Interconnections With Improved Reliability}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JIOT.2023.3267657}
  }
```

**完整代码 Complete code**
```python
import numpy as np
from cma import CMAEvolutionStrategy

def CRP_gen_APUF(num = 12000, stages = 64, xor_num = 1, seed = None):
    '''num    : CRP 个数；
       stages : APUF 的级数，即challenge的bit数
       xor_num: APUF 的个数，PUF 的最终响应为所有APUF响应的xor
       seed   : 生成随机CRP所用的种子, 若为None, 则每次生成的CRP将不相同
       '''
    np.random.seed(seed)
    # 生成 challenge
    chal = np.random.choice([0, 1], [num, stages]).astype(np.int8)
    # 生成延时参数
    para_delay = np.random.randn(xor_num, 2, stages)
    # 创建用于保存response的变量
    resp_tot = np.zeros([xor_num, num], dtype = np.int8)
    resp = np.zeros(num, dtype = np.int8)

    for apuf_inx in range(xor_num):
        para = para_delay[apuf_inx]

        # 根据APUF的结构计算两条路径的总延时差值
        delay_sum = np.zeros(num)
        for i in range(stages):
            para_select = para[chal[:, i], i]
            delay_sum = delay_sum * (1 - 2*chal[:, i]) + para_select

        # 获得response
        resp_tot[apuf_inx, delay_sum >0] = 1
        # XOR 运算
        resp = resp ^ resp_tot[apuf_inx]

    return chal, resp, para_delay, resp_tot

def fitness_fun(weight, chal, resp, xor_num):
    ''' chal和resp是正确的CRP。 
    利用weight和APUF的数学模型预测 response，返回预测错误率'''
    stage_num=chal.shape[1]
    crp_num  =chal.shape[0]

    # 根据数学模型，每个APUF需要 stage_num +1 个参数
    if weight.shape[0] != (xor_num * (stage_num +1)):
        print('Warning: weight.shape[0] != (xor_num * (stage_num +1)):')
    delay_para = weight.reshape(stage_num+1, xor_num)

    C =1 - 2*chal # 将challenge转换为 1, -1

    # 根据数学模型，将challenge 进行转换，以避免非线性运算
    C_linear =np.ones((crp_num, stage_num + 1))
    for i in range(stage_num-1, -1, -1):
        C_linear[:,i] = C_linear[:, i+1] * C[:, i]
    
    delay_sum = np.matmul(C_linear, delay_para)  # 矩阵相乘，计算最终的延时差值
    sum_prod = np.prod(delay_sum, axis = 1)  # 等价于 XOR operation

    resp_predict = np.zeros(crp_num).astype(np.int8)
    resp_predict[sum_prod > 0] = 1

    err_rate = np.sum(resp ^ resp_predict) / crp_num * 100
    return err_rate

train_num = 2000
test_num  = 2000
stage_num = 64
xor_num   = 1  # 1-> APUF， 2 -> 2XOR PUF， n -> nXOR PUF
train_iterations = 200 # 设置迭代次数

crp_num   = train_num + test_num

############ 生成测试集和训练集 #############
res = CRP_gen_APUF(crp_num, stage_num, xor_num, 10)
print('Uniformity is ', sum(res[1])/crp_num)
chal_train = res[0][:train_num]
resp_train = res[1][:train_num]
chal_test  = res[0][train_num:]
resp_test  = res[1][train_num:]

########## 用CMA-ES算法训练 ################
weight_size = (stage_num + 1) * xor_num
# 创建 CMA-ES类
es = CMAEvolutionStrategy(weight_size * [0], 1)
# 开始训练。注意：fitness_fun的weight参数由 optimize函数自动传入，args只需传递其它参数即可
es.optimize(fitness_fun, iterations = train_iterations, verb_disp=10, args = (chal_train, resp_train, xor_num))
res = es.result

########## 用测试集测试结果 #################
error_rate = fitness_fun(res[0], chal_test, resp_test, xor_num)
print('Prediction Accuracy is: ', 100 - error_rate, '%')
```
