# 实验一：文本分类

## 1. 项目说明

本项目使用 TF-IDF 将新闻文本转换为数值特征，并比较逻辑回归和线性支持向量机在新闻文本十分类任务上的性能。

训练数据包含 7368 条有标签文本，测试数据包含 2457 条无标签文本。训练数据按照 8:2 划分为训练集和验证集，并使用分层划分保持类别比例。

## 2. 文件说明

- `main.py`：完整实验代码
- `train_data.csv`：有标签训练数据
- `test_data_unlabeled.csv`：无标签测试数据
- `predictions_best.csv`：最终测试集预测结果
- `all_tuning_results.csv`：模型和参数对比结果
- `feature_tuning_results.csv`：TF-IDF 特征配置对比结果
- `confusion_matrix_best.png`：最佳模型混淆矩阵
- `confusion_pairs_best.csv`：最佳模型的类别混淆统计
- `validation_errors_best.csv`：最佳模型的验证集错误样本

## 3. 环境要求

实验使用 Python 3.10，主要依赖：

- numpy
- pandas
- scikit-learn
- matplotlib

可使用 Conda 创建环境：

```bash
conda create -n project1 python=3.10
conda activate project1
conda install numpy pandas scikit-learn matplotlib

4. 运行方法
确保以下文件位于同一目录：
- main.py
- train_data.csv
- test_data_unlabeled.csv
进入该目录后运行：
python main.py
程序将依次完成：
1. 加载并检查数据；
2. 按照 8:2 分层划分训练集和验证集；
3. 提取 TF-IDF 特征；
4. 调整逻辑回归和线性 SVM 的参数；
5. 比较不同特征配置；
6. 生成分类报告、混淆矩阵及错误分析；
7. 使用全部有标签数据训练最终模型；
8. 生成测试集预测文件。
5. 实验设置
- 随机种子：42
- 验证集比例：20%
- 划分方式：分层随机划分
- 主要评价指标：准确率
- 辅助评价指标：Macro-F1
- 比较模型：
  - Logistic Regression
  - Linear SVM
- 调整参数：
  - C = 0.1、1、10
- TF-IDF 特征实验：
  - unigram，5000维
  - unigram + bigram，5000维
  - unigram，10000维
  - unigram + bigram，10000维
TF-IDF 仅在训练子集上拟合，再使用相同的转换规则处理验证集，从而避免数据泄露。

6. 主要实验结果
在 5000 维 unigram 特征下，逻辑回归 C=10 的验证集准确率为 0.9132，Macro-F1 为 0.9135。
在同样的特征下，线性 SVM C=10 的验证集准确率为 0.9077，Macro-F1 为 0.9085。逻辑回归的性能略高，而且训练速度明显更快。
进一步将 unigram 特征数量增加到 10000 后，逻辑回归的验证集准确率提高到 0.9261，Macro-F1 提高到 0.9265。
加入 bigram 没有进一步改善验证性能，并明显增加了特征提取时间。

7. 最终模型
最终采用以下配置：
- 特征表示：TF-IDF
- ngram_range：(1, 1)
- max_features：10000
- 分类器：Logistic Regression
- C：10
- max_iter：1000
- random_state：42
确定配置后，使用全部 7368 条有标签数据重新训练模型，并预测 2457 条测试文本。
最终预测保存在：
predictions_best.csv
该文件没有表头和索引，每行包含一个预测类别。

8. 错误分析
最佳模型在 1474 条验证数据中错误分类 109 条。
主要混淆包括：
- 类别 7 被预测为类别 2：12 条
- 类别 1 被预测为类别 2：9 条
- 类别 2 被预测为类别 7：9 条
- 类别 2 被预测为类别 0：6 条
- 类别 2 被预测为类别 1：5 条
人工检查发现，部分错误文本篇幅较短，主题信息不足；数据中还包含邮件地址、组织名称和引用信息等内容，可能对文本特征造成干扰。

9. 复现说明
程序使用固定随机种子 42。相同版本的数据和依赖环境下，数据划分及主要模型结果应当可以复现。不同机器上的运行时间可能有所差异。
