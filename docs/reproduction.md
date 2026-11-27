# 复现 climate-claim-verifier

## 环境与输入

在仓库根目录运行。使用 Python 3.11，完整数据训练建议使用 CUDA GPU。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
climate-claim-verifier setup-nltk
```

通过 `--data-dir` 指定数据目录。默认是 `data/`，需要以下四个文件。

| 文件 | 内容 |
| --- | --- |
| `evidence.json` | 证据 ID 到段落文本的映射 |
| `train-claims.json` | 训练声明、标签及证据 ID |
| `dev-claims.json` | 开发集声明、标签及证据 ID |
| `test-claims-unlabelled.json` | 无标签测试声明 |

### 输入格式

`evidence.json` 使用证据 ID 作为键、段落文本作为值。

```json
{
  "evidence-0001": "Multiple surveys report a high level of scientific consensus on human-caused warming."
}
```

训练集和开发集以声明 ID 为键，包含声明文本、四分类标签和标注证据 ID。

```json
{
  "claim-0001": {
    "claim_text": "Most climate scientists agree that humans are causing recent warming.",
    "claim_label": "SUPPORTS",
    "evidences": ["evidence-0001"]
  }
}
```

测试集使用相同的声明 ID 和文本结构，仅包含 `claim_text`。标签映射固定为 `SUPPORTS=0`、`REFUTES=1`、`NOT_ENOUGH_INFO=2`、`DISPUTED=3`。标注和预测中的证据 ID 均对应 `evidence.json` 的键。

输入数据保存在本地，默认的 `data/` 目录已加入 Git 忽略规则。完整数据及已训练的检查点需另行准备。

## 运行流程

首次运行需要下载预训练模型。检索阶段构建索引、训练召回与重排模型并导出证据，分类阶段读取证据并训练分类器。两个阶段使用相同的输出目录。

```bash
climate-claim-verifier retrieve --data-dir data --output-dir outputs/run
climate-claim-verifier classify --data-dir data --output-dir outputs/run
```

可以用 `climate-claim-verifier run` 按顺序执行这两步。所有命令也可以用 `python -m climate_verification` 调用。

- 检索时加 `--include-rrf` 可以导出 RRF 证据，供 `classify --all-experiments` 运行四套端到端方案。
- 检索时加 `--component-ablations` 可以对照 BM25、dense、RRF 和关闭疑似假负例过滤的重排器。
- 当数据和检索输出相同时，分类可加 `--reuse-classifiers` 复用输出目录里的检查点。

检索结果写入 `end_to_end_predictions/`，分类输入写入 `classifier_inputs/`。模型和训练摘要位于 `retrieval_models/` 和 `end_to_end_classifiers/`。

默认方案返回的测试集预测位于 `outputs/run/end_to_end_outputs/max_gap_ensemble_test_output.json`。每个结果包含 `claim_text`、`claim_label` 和按选取顺序排列的 `evidences`。

## 查看与评分

仓库中的 [开发集预测](../predictions/end_to_end_outputs/max_gap_ensemble_dev_prediction.json) 支持直接查看，评分只需预测文件和开发集标签。

```bash
climate-claim-verifier inspect claim-2580 \
  --predictions predictions/end_to_end_outputs/max_gap_ensemble_dev_prediction.json

climate-claim-verifier evaluate \
  --predictions predictions/end_to_end_outputs/max_gap_ensemble_dev_prediction.json \
  --groundtruth data/dev-claims.json
```

四套端到端方案的分数、开发集预测和测试集输出位于 [实验结果目录](../predictions/end_to_end_outputs/)。

`*.ensemble.json` 记录集成成员、权重和单模型开发集准确率。`end_to_end_classifiers/` 下的 `summary.json` 记录训练参数、损失和检查点路径，模型权重需单独准备。

## 数据与评分

已有实验使用固定划分的气候声明核查数据，包含 1,208,827 条证据段落、1,228 条训练声明、154 条开发集声明和 153 条无标签测试声明。复现已有分数时需要使用相同的数据划分、证据 ID 和标签映射。

评分实现位于 [evaluation.py](../climate_verification/evaluation.py)。证据 F1 按声明计算后取平均，分类准确率为正确判定的声明比例，综合分数为二者的调和平均。
