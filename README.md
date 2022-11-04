## 分步复现wechat大数据竞赛2021 top1方案
### 方案亮点
  1、对比了不同embedding初始化方法的效果

  2、利用DIN对序列特征进行处理

### 模型效果
#### **版本1**
- 特征
   - user id
   - feed id
   - w2v同现embedding初始化feed id embedding 1
   - feed embedding初始化feed id embedding 2
- 网络
   - MMoE 
- 设置
   - epoch_num: 5
   - expert_num: 8
- AUC
   - 'score': 0.549608
   - 'score_detail': 
     - 'read_comment': 0.532772
     - 'like': 0.534525
     - 'click_avatar': 0.600776
     - 'forward': 0.559866
#### **版本2**
- 特征
   - 利用全连接层融合feed emb 1和feed emb 2
   - 增加用户多行为历史列表
   - 增加用户单行为历史列表
   - 增加用户历史完成度列表
   - 增加展现但未交互历史队列
- 网络
   - DLRM：DIN+MMoE+cos交互网络
- 设置
   - epoch_num: 5
   - expert_num: 5
- AUC
   - 'score': 
   - 'score_detail': 
     - 'read_comment': 
     - 'like':
     - 'click_avatar': 
     - 'forward':

#### **版本3**
- 特征
   - 增加context信息
   - 使用w2v同现emb初始化用户embedding
   - 增加side information
- AUC
   - 'score': 
   - 'score_detail': 
     - 'read_comment':
     - 'like':
     - 'click_avatar': 
     - 'forward': 
     -