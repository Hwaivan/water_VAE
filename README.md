# 水声波形 VAE 预训练

独立工作区，用干净的单声道 WAV 训练连续潜变量 VAE。提供预训练、确定性重建评估、单文件 WAV 重建和潜变量导出。运行时只需本目录和依赖环境。

## 安装与快速验证

建议 Python 3.10–3.12。用训练机器对应的 PyTorch 环境安装以下依赖。CPU 可运行全部入口，CUDA 环境可用单卡和混合精度。

```bash
cd vae_workspace
python -m pip install -r requirements.txt
python smoke.py --device cpu
python -m unittest discover -s tests -v
```

冒烟测试自动生成多频 WAV 与临时配置，验证两次更新、续训到四次更新、评估、WAV 重建和潜变量导出。CPU 路径还逐项比较续训和连续训练的模型权重。可用 `--output-dir runs/smoke_check` 保留测试产物。CUDA 环境可运行 `python smoke.py --device cuda`。

## 数据

配置 `data.train_manifest`、`data.valid_manifest`、`data.test_manifest`。按原始录音分组划分训练、验证和测试，避免同一录音的切片或混合版本跨集合。训练需要 train/valid，默认独立评估读取 test。仓库中的正式配置提供 manifest 路径约定，需填入自己的列表。

```text
# data.manifest_mode: clean
clean/recording_01.wav
clean/recording_02.wav
```

```text
# data.manifest_mode: paired（两列以 TAB 分隔）
noisy/mix_01.wav	clean/recording_01.wav
noisy/mix_02.wav	clean/recording_02.wav
```

- 支持 UTF-8 BOM、注释和空行。配对行以 TAB 分列，路径无空格时也可用空白分隔。只读取第二列 clean WAV。ID 取第一列 basename 去掉最后一个扩展名，区分大小写，每个 manifest 内必须唯一。
- YAML 中的数据、类别 TSV 和输出路径相对于 `vae_workspace/`。音频相对路径以 manifest 所在目录为基准。若旧 DiT manifest 中的相对路径以 `dit_workspace/` 为基准，设置 `data.audio_root: ../dit_workspace`。直接运行 Python 时，命令行路径相对于当前目录。Bash 包装脚本会先进入 `vae_workspace/`，也接受绝对路径。
- 读取 PCM 或浮点 WAV，声道取均值，以带抗混叠的多相滤波重采样至默认 16 kHz。PCM 转为浮点满量程单位，保留录音之间的相对幅值。训练随机裁剪，验证和测试默认取居中片段。短音频保留有效长度，批次右侧补零，进入卷积前按有效长度分组去除批次填充。模型再按下采样倍率做最少的右侧零填充。
- 可选类别 TSV 使用 `file_id<TAB>class_id`，首行可为同名表头。类别从 0 开始，配置 `model.num_classes` 后可在 `data.class_labels.train/valid/test` 指定文件。提供 TSV 时，该集合每个 manifest ID 都必须有标签。缺失、重复和越界均报错。未配置 TSV 的集合可以直接训练或评估。paired 模式使用 noisy ID，例如 `mix_01`，clean 模式使用 clean ID，例如 `recording_01`。

## 模型和配置

`WaveformVAE` 使用 Conv1d、GroupNorm 残差块、步幅下采样和对称的转置卷积。编码器输出 `mu, logvar`，训练通过重参数化采样 `z`。评估、推理和导出使用 `mu`。步幅卷积和转置卷积采用同一 padding，解码只裁右端的对齐填充。

```python
model.eval()
mu, logvar = model.encode(waveform)       # 输入 [B,T] 或 [B,1,T]
restored = model.decode(mu, waveform.shape[-1])
result = model(waveform)                  # waveform, mu, logvar, z, logits
```

潜变量布局为 `[B,D,ceil(T/S)]`，其中 `D=latent_channels`，`S=prod(strides)`。默认步幅 `[4,4,4,2]` 给出 128× 时间下采样和 125 帧/秒。5 秒输入 `[B,80000]` 对应 `[B,64,625]`，每条从 80,000 个波形标量变为 40,000 个 latent 标量。默认模型有 528,289 个可训练参数。独立调用 `decode(z)` 返回对齐后的全长。传入编码前长度可恢复原始样本数。

| YAML | 用途 |
| --- | --- |
| `experiment` | 名称、输出目录、随机种子 |
| `data.segment_seconds` | 训练裁剪长度。设为 `null` 读取全长 |
| `data.segment_validation` | 启用验证和测试的居中裁剪 |
| `data.max_full_seconds` | 全长评估、推理、导出的时长上限，默认 60 秒 |
| `model.channels/strides/latent_channels` | 卷积宽度、下采样倍率、latent 通道数 |
| `loss.waveform_weight/mrstft_weight` | 波形 L1 和 MR-STFT 权重 |
| `loss.kl_weight/kl_warmup_steps` | 每 latent 元素的平均对角高斯 KL 权重，按优化器更新数线性升温。步数 0 使用固定 β |
| `model.classification_head/loss.classification_weight` | 默认关闭的时序均值池化 + Linear 辅助分类。启用时设置类别数及 CE 权重，仅在有标签的样本上计算 |
| `training` | AdamW 学习率、batch size、epoch、梯度裁剪、CUDA AMP |
| `metrics/evaluation` | 低能量阈值、KL 塌缩提示阈值、保存的 WAV 示例数 |

MR-STFT 对各分辨率的 spectral convergence 和自然对数幅度 L1 求平均。第一个分辨率与本地 DiT Flow 配置一致，使用周期 Hann、`512/400/100`（FFT/窗长/步长）、`center=true`、`normalized=false`、`onesided=true`。短到无法反射填充时用零填充。STFT 与损失统计以 float32 计算。参考频谱能量极低时，重建损失保留波形 L1 和对数幅度项，并省略对应分辨率的 spectral convergence。日志记录 VAE loss、单独的分类 CE、KL、posterior mean 的均值和 RMS、posterior std 均值，以及基于验证 KL 阈值的塌缩提示。提示可与这些统计量和重建曲线一起看。

本地 DiT 使用 complex RI 或压缩幅度特征。幅度路径用 noisy phase 与 iSTFT 重建。该本地版本有 `dit.data`，未找到 `mel.py` 和 `class_labels.py`。VAE 复用 STFT 参数和 paired manifest ID 约定，使用独立数据和类别读取实现。

## 训练和续训

```bash
python train.py --config configs/vae_waveform_16k_5s.yaml
GPU_ID=0 DEVICE=cuda bash run_train_vae.sh

python train.py --config runs/vae_waveform_16k_5s/runtime_config.yaml \
  --resume runs/vae_waveform_16k_5s/checkpoints/last.pt --epochs 60
```

输出包含 `runtime_config.yaml`、`metrics.jsonl`、`logs/train.log`、`reconstructions/epoch_*/` 和 `checkpoints/{best,last}.pt`。每个 epoch 结束后保存，`best.pt` 按验证的加权重建损失（波形 + MR-STFT）选择。checkpoint 含模型、AdamW、GradScaler、随机状态、配置、epoch、更新数与最优指标。原目录续训使用 `last.pt`，从下一个 epoch 开始。`--epochs` 是累计目标；配置中的数据、模型、损失、种子和优化器参数需保持一致。也可用 `--output-dir` 续到新目录；若历史 `best.pt` 在源 checkpoint 同目录且匹配，则一并继承。训练脚本支持顶部变量及环境变量 `GPU_ID`、`DEVICE`、`CONFIG_FILE`、`EXPERIMENT_NAME`、`RESUME`、`PYTHON`。三个 Bash 包装脚本均从自身目录启动，支持额外 CLI 参数。

## 重建评估与单文件推理

```bash
python evaluate.py --config configs/vae_waveform_16k_5s.yaml \
  --checkpoint runs/vae_waveform_16k_5s/checkpoints/best.pt --split test

python infer.py --config configs/vae_waveform_16k_5s.yaml \
  --checkpoint runs/vae_waveform_16k_5s/checkpoints/best.pt \
  --input path/to/input.wav --output path/to/reconstructed.wav
```

评估写出逐文件 `per_file.jsonl`、汇总 `summary.json` 和有限数量的 WAV 示例。指标包括 waveform L1、SDR（参考功率/误差功率）、去均值 SI-SNR、第一个 STFT 分辨率的 spectral convergence、逐帧频率维 dB RMS 再对帧求均值的 log-spectral distance，以及 KL。静音、低能量、损坏或非有限样本会被标记并排除在汇总均值之外。SI-SNR 单独报告有效计数，近常量参考的 SI-SNR 为 null，近常量重建为 −80 dB。可用 `--manifest` 覆盖当前集合列表，或用 `--full-audio` 进行全长评估。输出默认在 `runs/<experiment>/evaluation/<split>/`，可用 `--output-dir` 指定新目录。

推理整段编码一次，在模型采样率完成重建，再恢复输入采样率与精确采样点数，写出单声道 float32 WAV。输入超过 `data.max_full_seconds` 会明确报错。推理只需要模型权重、配置和输入 WAV。

## 潜变量导出

```bash
python export_latents.py --config configs/vae_waveform_16k_5s.yaml \
  --checkpoint runs/vae_waveform_16k_5s/checkpoints/best.pt \
  --manifest path/to/clean.list --output-dir runs/exported_latents

python export_latents.py --config configs/vae_waveform_16k_5s.yaml \
  --checkpoint runs/vae_waveform_16k_5s/checkpoints/best.pt \
  --fit-train-stats --output-dir runs/train_latents
```

每个文件生成一个 `.pt`，含 `mu [D,T_latent]`、`file_id`、模型采样率下的有效波形长度、源采样率和长度、latent shape、下采样倍率、checkpoint SHA-256、版本、epoch 和更新步数。`index.jsonl` 记录索引和同样的元数据。导出使用全长音频及相同的全长限制。用 `torch.load(path, weights_only=True)` 读取。

`--fit-train-stats` 独占导出模式，读取 checkpoint 所保存的训练 manifest 路径，写出 `latent_stats.json`。在可用的验证、测试清单中发现同一 clean 路径时会报错。训练 manifest 内容需与训练时一致。统计以全长训练音频的 latent 帧为权重，按通道估计总体均值和标准差。后续训练、验证、测试统一使用训练集的 `mean` 和 `normalization_std`（标准差下限为 1e-6）。

## 验证范围与后续接口

本轮验证范围是 VAE 自编码重建流程。真实水声数据的重建质量、降噪收益、下游分类收益和 CUDA 训练尚未验证。分类辅助头用于训练和诊断，独立下游分类器需要另行评估。

后续可冻结本 VAE，用 clean posterior mean 和训练集统计量定义 Latent Flow-DiT 的目标空间，用独立 noisy condition encoder 提供条件。生成 latent 先逆标准化，再将其与模型采样率下的有效长度传给 `decode`。接入冻结水声识别教师时，需要先确定具体教师、采样率、输入预处理和目标特征层，再用对齐的 clean WAV 作为教师目标，对重建 WAV 的对应层特征计算蒸馏损失。本工作区实现范围止于 VAE 预训练、辅助分类、重建及 latent 导出。
