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
