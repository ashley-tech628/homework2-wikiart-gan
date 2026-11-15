# Homework 2: full WikiArt / pretrained StyleGAN2 transfer

这套代码替换之前的自定义 GAN。它采用 **NVIDIA 官方 StyleGAN2 网络 + Peter Baylies 公开 WikiArt 预训练权重 + 在指定完整数据集上的微调**。不是从零训练，也不是把图片做锐化、超分辨率或使用其他图像生成服务。请在报告中明确注明迁移学习及来源。

## DataHub：运行一条命令

把 `datahub_stylegan2_wikiart_code.zip` 上传到 DataHub 的 home。在 Terminal 执行：

```bash
cd ~
unzip -o datahub_stylegan2_wikiart_code.zip
cd datahub_stylegan2_wikiart
bash run_datahub.sh
```

程序自动完成：GPU/PyTorch 检查、权重下载与校验、原始完整数据下载或复用、训练前预览、全数据索引、初始验证指标、训练、定时断点、验证指标选择、结果打包。

完整数据、权重下载、Python 环境和缓存都在 `/tmp`。优先复用之前的 `/tmp/homework2-full-wikiart-$USER/raw`；不会把旧的 128px 图片放大充当高分辨率原图。**不需要先将 8 万张图片逐张转存**；训练时读取原始图片，按需中心裁剪和缩放。

若原始数据放在其他位置：

```bash
WIKIART_SOURCE=/tmp/你的原始WikiArt路径 bash run_datahub.sh
```

想先看公开模型的起点、暂不下载数据或训练：

```bash
bash run_datahub.sh preview
```

## 运行时间与中断恢复

- 默认保留完整 **1024×1024** 网络。不是把 128px 输出放大，所有独立 PNG 都是模型原生输出。
- 默认有效 batch=16，GPU microbatch=2，R1/路径长度正则的估计 batch=1；针对 24GB GPU。第一次完整优化步骤会测试正则、梯度、优化器以及完整断点写入。没有在用户的 DataHub GPU 上远程实测；本地 RTX 5070 8GB 测试使用 microbatch=1。
- 默认目标 **80 kimg**（微调判别器共看到约 80,000 张训练图片，至少完整走过一轮 73,299 张训练集；精确人数以 manifest 为准）。这是初始实验预算，不是画质保证，也不是声称最佳训练长度。
- **每次运行约 2.5 小时后主动保存并返回**，不是失败。停止检查在一个完整优化步骤之后；最后验证/导出会额外花时间。初始化/下载另需时间。根据日志实测速度，完成目标可能需要多个会话，不能承诺几小时完成。
- 同一命令再次运行，会从 `~/homework2-stylegan2-wikiart/latest.pt` 恢复，包括 G、D、EMA、两个 Adam 优化器、随机数状态、训练位置、ADA 概率和路径长度统计。不会从零开始。每 10 分钟另存一次，服务器突然终止时最多丢失最近一次成功保存后的进度；单步或存盘太慢时这个窗口可能延长。
- `/tmp` 被清空时，数据/环境需要重新下载，但 home 里的断点保留。**home 需要约 2 GiB 可用配额**用于完整断点的原子替换；实测单个断点约 817 MiB。初始化和第一个完整训练步骤就验证能否写入，写不进去会报错停下。不要删除 home 的实验文件夹。
- `Ctrl+C` 请求在当前步骤结束后保存退出；不要连续强制结束进程。

查看状态：

```bash
cat ~/homework2-stylegan2-wikiart/status.json
```

`session_time_limit`：同一命令继续。`target_reached` 且 `all_training_images_seen=true`：完成本次完整数据预算。`quality_guard`：连续两次验证 FID 比初始模型差 25% 以上，程序停止；先检查 best 模型，不要盲目增加训练时间。

只在新实验第一次运行前修改 batch/microbatch/LR；恢复时程序检查这些设置一致。不同会话可改变单次时长/总预算：

```bash
HW2_MAX_HOURS=4 bash run_datahub.sh
# 确认验证表现值得继续之后，才增加总预算：
HW2_KIMG=160 bash run_datahub.sh
```

## 去哪里看结果

DataHub 文件浏览器打开 `homework2-stylegan2-wikiart`：

| 文件 | 内容 |
|---|---|
| `pretrained_before_finetuning_psi1.png` | 公开预训练模型起点；不是你的微调成果 |
| `latest_grid_psi1.png` | 当前模型、固定连续 16 个种子、无截断的预览 |
| `partial_grid_psi1.png` | 尚未完整遍历训练集时的最佳阶段预览，不能称为已完成全数据训练 |
| `best_grid_psi1.png` | 已完整遍历训练集至少一轮的候选中，验证 FID 最佳的**微调**模型，主展示图 |
| `best_grid_psi1_1024px/` | 对应的原生 1024px 单张图片 |
| `best_grid_psi07.png` | 同一组种子，截断 psi=0.7 的另附展示；可能牺牲多样性 |
| `loss_curves.png` / `losses.csv` | 损失和判别器分数 |
| `validation.jsonl` | 初始模型及各次微调验证指标 |
| `pretrained_test_metrics.json` | 同协议下的公开预训练起点测试指标 |
| `finetuned_test_metrics.json` | best 微调模型的最终测试指标 |
| `latest.pt` | 完整恢复断点，不能用 `best_finetuned.pt` 替代 |
| `best_finetuned.pt` | 选出的 EMA 生成器，仅供生成/评估，不含恢复训练的优化器 |
| `best_partial.pt` | 尚未完成一轮时的临时模型；不会替代最终完整数据模型 |

每次正常退出都会生成 **`~/homework2-stylegan2-results.zip`**，下载这个 ZIP 可以查看/撰写报告。它不包括大权重；若要离线备份训练，请另行下载 `latest.pt`。首次训练会先生成预览，随后可能下载较大的原始数据。

## 模型、数据与评估

- 官方 StyleGAN2：调制/解调卷积、映射网络、skip RGB、残差判别器、minibatch standard deviation。
- 公开无条件 WikiArt 1024px 模型初始化 G（使用其 EMA）及 D；保留完整网络。全部可训练权重参与微调。Adam LR=1e-4、betas=(0, .99)，按 lazy regularization 比例修正；非饱和 logistic GAN 损失；R1 gamma=10；path length weight=2；每16步正则；EMA 半衰期10kimg。
- NVIDIA ADA 的颜色变换子集 + 水平翻转。目标 real-sign=.6、调整时间常数500kimg；不是原版全部几何 ADA 配置。为减少计算，R1 和 path-length 各用一个随机样本估计，正则期望不变、方差会更高。正则使用 FP32，其他阶段使用模型自带混合精度。
- 默认算子使用 NVIDIA 自带的 PyTorch reference 实现，**无需 nvcc 编译**，所以不依赖 DataHub 的旧 CUDA12.0 编译器。这个选择可能比正确配置的融合 CUDA 内核慢；优先保证可运行性和数值验证。上游源码、许可证、固定 commit 和修改说明都包含在 ZIP 中。
- 严格检查原始数据 **81,444 张、27 风格目录**。固定 seed2026，90%训练，剩余一半验证、一半测试。不对训练集截取小子集；每轮按确定性顺序洗牌遍历全部训练图片。损坏文件明确报错，不偷偷跳过。所有风格共同训练一个无条件生成器，不按风格标签控制输出。
- 验证512对用于选择 `best_finetuned.pt`，只有完整遍历训练集至少一轮的候选才有资格成为最终模型；之前另存 `best_partial.pt`。初始公开模型单独保存，不冒充“微调最好”的结果。每10kimg和每次正常退出验证一次。默认完整训练预算结束后才用测试1000对评估初始/微调模型；同数量、同种子、同缩放。
- 使用 `pytorch-ignite==0.5.4`：主 FID 为 Inception pool3 2048维；另存博客默认1000维概率空间的 FID 和 IS。评估 psi=1、随机噪声，不挑图；psi=.7 只作另附展示。
- **公开预训练模型也用过 WikiArt，和这里的验证/测试图片可能重叠**。划分只保证这次微调未见测试图片；不能宣称对预训练阶段完全未见数据的泛化。不要将不同样本数、分辨率或实现的 FID 直接当成严格比较。
- 自动选模型依赖有限样本 FID，有噪声，也不等价于人眼质量或多样性保证。仍需查看固定种子网格，检查重复和伪影。长训练并不保证改善；真实结果必须来自你的微调运行。

完成至少一轮后，可用 `bash run_datahub.sh evaluate` 单独评估当前已选完整数据模型；报告应注明实际 nimg/完整遍历次数。尚未完成一轮时，先看阶段预览和验证指标。默认流程不会拿测试集反复挑选模型。

## 公开来源和作业说明

- NVIDIA 官方代码：https://github.com/NVlabs/stylegan2-ada-pytorch
- CUDA/torch 适配文件来源及 commit：`UPSTREAM.json`
- Peter Baylies WikiArt 模型：https://archive.org/details/wikiart-stylegan2-conditional-model
- 下载文件：`WikiArt_Uncond2.pkl`，SHA-1 `48634616ab4b4408b28378b649da67a52c309ea9`（与公开 archive 元数据比对），下载后严格校验；专用受限反序列化器只允许数组和旧网络容器，然后转换为可 `weights_only=True` 读取的 state dict。
- 指定数据：https://www.kaggle.com/datasets/steubk/wikiart
- Ignite 作业参考：https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/

报告需要说明使用官方实现与公开预训练权重，以及本项目的数据、训练/恢复、评估改动。不要宣称从零独立实现了 NVIDIA 网络。作业写了“implement a GAN”；原先自己实现并完成训练的全数据 baseline 可保留作基础实验，本方案作为明确注明的迁移学习改进。仅靠此方案不能保证老师对“独立实现”的要求或满分。

保留从零初始化入口（不读取预训练张量）：

```bash
python init_scratch.py --output /tmp/wikiart-scratch-initial.pt
```

然后给 `train.py --initial` 指定该文件，并使用**新的** `--persist` 实验目录；其他参数参考 `run_datahub.sh`。同架构从零训练需要远高于80kimg的预算，这不是推荐的短期高画质路径。

本项目及上游代码仅供研究/评估，遵循随包 NVIDIA 许可证。`VALIDATION.md` 记录交付前实际执行的检查及其局限。
