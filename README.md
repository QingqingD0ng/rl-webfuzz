# 强化学习Webfuzz API 文档

## 启动服务
首先需要安装依赖
```
pip install -r requirements.txt
```
启动Flask应用
```
python app.py
```
启动成功后服务器地址为http://127.0.0.1:5000/ ，服务器日志保存在logs/目录下。
## API说明
API包括模型初始化、参数设置、预测action、模型训练等。模型有PolicyGradient、DQN、DDQN三种。
### /initDQN 
初始化DQN 或 DDQN 模型
#### POST
|参数         |必填|默认值|类型 |描述      |
|-------------|----|------|-----|---------------|
|n_actions    |是  |无    |int  |action的总个数 |
|ddqn         |否  |false |bool |值为false时模型是dqn，值为true时是ddqn|
|target_update|否  |10    |int  |仅在ddqn模型下需要，ddqn中两个网络经历target_update个回合进行同步|
|max_len      |否  |512   |int  |URL的最大长度，超过长度会被截取|
|episode      |否  |10000 |int  |训练回合数，次数越多耗时越长，效果也可能更好，一次回合代表从初始种子到下一次初始种子的过程。|
|batch_size   |否  |32    |int  |一次训练的样本数量，一般选择32或64，也可尝试用action的总个数|
|memory_size  |否  |1000  |int  |神经网络的记忆大小，能记住多少步内的动作，旧的记忆会被新的替代|
|EPSILON      |否  |0.1   |float|模型的探索倾向，每次action将有EPSILON的概率随机进行选择|
|GAMMA        |否  |0.95  |float|对未来奖励的衰减率，值越大未来的奖励值越重要，值越小当前的奖励值越重要|
|learning_rate|否  |0.01  |float|学习率，模型的学习速度，一般可在0.1、0.001和0.0001中选择，值越大模型更新越快，但有可能陷入局部最优。值越小，消耗的时间越大，需要增加episode的数量。|

#### request sample
```json
{n_actions: 64, ddqn: true,max_len:1000}
```

#### return
```json
{code: 400, msg: "Error!"}
{code: 200, msg: "Dqn initiated successfully!", data: 模型参数}
```
### /initPolicyGradient
初始化Policy Gradient模型
#### POST
|参数         |必填|默认值|类型 |描述      |
|-------------|----|------|-----|---------------|
|n_actions    |是  |无    |int  |action的总个数 |
|max_len      |否  |512   |int  |URL的最大长度，超过长度会被截取|
|episode      |否  |10000 |int  |训练回合数，次数越多耗时越长，效果也可能更好，一次回合代表从初始种子到下一次初始种子的过程。|
|GAMMA        |否  |0.95  |float|对未来奖励的衰减率，值越大未来的奖励值越重要，值越小当前的奖励值越重要|
|learning_rate|否  |0.01  |float|学习率，模型的学习速度，一般可在0.1、0.001和0.0001中选择，值越大模型更新越快，但有可能陷入局部最优。值越小，消耗的时间越大，需要增加episode的数量。|

#### request example
```json
{n_actions: 64, max_len:1000,episode:100000}
```

#### return example
```json
{code: 400, msg: "错误的原因和提示"}
{code: 200, msg: "PolicyGradient initiated successfully!", data: 模型参数}
```

### /startTraining
开始训练模型，必须在已初始化模型后使用。
#### POST
|参数         |必填|默认值                           |类型|描述  |
|-------------|----|---------------------------------|----|------|
|initial_state|是  |无                               |str |初始URL|
|save_path    |否  |./checkpoints/%Y-%m-%d-%H.%M.%S/ |str |模型保存的文件夹位置，注意末尾要加上“\”符号，建议不要修改|

#### request example
```json
{initial_state: "http://127.0.0.1:8888/notebooks/server.ipynb#"}
```

#### return example
请求成功后，本地save_path文件夹会出现一个meta.txt，记录了该模型的参数信息。
```json
{code: 400, msg: "错误的原因和提示"}
{code: 200, msg: "Get action successfully!",data:模型参数}
```   

### /action
获取模型预测的action
#### POST
|参数         |必填|默认值|类型 |描述   |
|-------------|----|------|-----|-------|
|state        |是  |无    |str  |当前URL|

#### request example
```json
{state: "http://127.0.0.1:8888/notebooks/server.ipynb#"}
```

#### return example
返回action序号n，int类型，n属于[0,n_actions)

```json
{code: 400, msg: "错误的原因和提示"}
{code: 200, msg: "Get action successfully!", data: 1}
```

### trainStep

一个训练step,如果当前回合已达到设置的episodes，则不会训练。模型训练时会自动记录reward历史，保存至当前save_path里的XXXloss_history.py。训练过程中如果效果提升时，模型会自动保存。

#### POST
|参数      |必填|默认值|类型  |描述  |
|----------|----|------|------|------|
|state     |是  |无    |str   |做出action之前的state|
|action    |是  |无    |int   |选取的action序号|
|next_state|是  |无    |str   |做出action之后的state|
|reward    |是  |无    |float |奖励值，属于[-1,1]，目前采取的是如果没找到flag，奖励值为-0.1，找到的话奖励值为1|
|done      |是  |无    |bool  |当前回合是否结束，结束为1，没结束为0。依据是否找到flag结束，为防止回合时间过长，也可设置一个最大变异次数的限制，超过最大变异次数结束。|

#### request example
```json
{state:"http://127.0.0.1:8888/notebooks/server.ipynb#",action:1,next_state:"http://127.0.0.1:8888/notebooks/server.ipynb#alert(1)",reward:-0.1,done:false}
```

#### return example
```json
{code: 404, msg: "已达到最大训练回合数，如果想继续训练请使用'/addEpisode'请求"}
{code: 400, msg: "错误的原因和提示"}
{code: 200, msg: "Train step successfully!"}
```

### /addEpisode
增加训练回合数，适用于已经达到模型最大训练次数，效果不够好，还想继续训练的情况。

#### POST
|参数      |必填|默认值|类型  |描述  |
|----------|----|------|------|------|
|episode   |是  |无    |int   |还需要增加的训练回合数|

#### request example
```json
{episode:100000}
```

#### return example
请求成功后，即可以继续训练。
```json
{code: 400, msg: "错误的原因和提示"}
{code: 200, msg: "Add episode successfully!"}
```

### /describeModel
描述当前模型的参数，忘记了当前模型的时候使用
#### GET
#### return example
```json
{code: 400, msg: "错误的原因和提示"}
{code: 200,msg:'Describe model successfully!',data:模型参数}
```

### /setParameter
设置模型参数，可以在模型已经初始化但还没有开始训练时，更改模型的参数。
#### POST
post需要改的参数名和值就可以了，可以修改的参数参考模型初始化

#### request example
```json
{ddqn:true,batch_size:64}
```
#### return example
```json
{code: 400, msg: "错误的原因和提示"}
{code: 200,msg:'Describe model successfully!',data:模型参数}
```

### /deleteModel
删除当前模型，想要重新训练或更换其他模型时使用
#### GET
#### return example
```json
{code: 200,msg:'Model deleted successfully!'}
```  
## 训练方法
### 模型初始化
想要训练模型，首先要选择一个模型并初始化，可以使用/initDQN 或 /initPolicyGradient请求，并设置好适当的参数，如果对深度学习调参不太了解建议使用默认参数。
```python
n_actions = 10000
episode = 50000
res = requests.post(url='http://127.0.0.1:5000/initDQN',json={"n_actions":n_actions,"episode":episode,"ddqn":False})
print(res.text)
```
### Fuzz之前
在开始进行Fuzz之前需要使用/startTraining请求，发送初始URL，初始化各个网络的神经元。
```python
#设置初始状态
initial_state = 'http://127.0.0.1:8888/edit/ReadMe.md'
res = requests.post(url='http://127.0.0.1:5000/startTraining',json={"initial_state":initial_state})
print(res.json())
```
### 模型训练
训练过程应该和Fuzz结合成一个循环,可以参考下面的伪代码。
```python
episode = 10000 #设置好训练的回合数
i= 0
state = initial_state
while True:
    #根据当前的状态state获取action
    res = requests.post(url='http://127.0.0.1:5000/action',json={"state":state})
    res = res.json()
    print(res)
    if res['code'] == 200:
        action = res['data']
    print('choosing action:',action)
    #把获得的action转换成新的state
    next_state = convert_state（state，action）
    #把变异后的state拿去fuzz，获得奖励reward和fuzz结果done
    reward, done = fuzz(new_state)
    #根据这一次变异训练一次trainStep
    data = {'state':state,'action':action,'next_state':next_state,'reward':reward,'done':done}
    res = requests.post(url='http://127.0.0.1:5000/trainStep',json=data)
    print(res.json)
    #回合结束
    if done:
        #重设初始状态
        state = initial_state
        i+=1
        #所有回合训练完毕，结束训练
        if i == episode:
            break
        #没达到设定的回合数，继续训练
        continue
    #更新当前state
    state = next_state
```
### 重新训练模型
重新训练模型首先要使用/deleteModel请求，之后再重复一次以上的过程即可。
```python
res = requests.get(url='http://127.0.0.1:5000/deleteModel')
print(res)
```