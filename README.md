# HiST-FEAST

病理图像到空间基因表达预测的实验项目。当前模型依次使用 **UNI → GCN → ResNet**，训练入口为 `train.py`。

本文汇总项目目录、训练时读取和生成的文件，以及当前脚本的路径规则。

## 项目根目录

当前本地项目目录：

```text
E:\workplace\HiST-FEAST
```

下文中的相对路径均相对于项目根目录。迁移项目后，根目录会随脚本位置变化，无需保留这个 Windows 盘符。

## 目录结构

```text
HiST-FEAST/
├── README.md                       # 项目结构和路径说明
├── train.py                        # 训练入口
├── evaluate.py                     # 使用已保存 checkpoint 独立评估 test 数据
├── GCN.ipynb                       # 实验 Notebook
├── .gitignore                      # 忽略本地 token 配置
├── config/
│   ├── token.json                  # 下载脚本使用的 Hugging Face token
│   └── train.json                  # 现有配置文件；当前 train.py 不读取它
├── download/
│   ├── HEST_dataset_download.py    # 下载 HEST 图像 patch 和表达数据
│   └── UNI_ckpt_download.py        # 下载 UNI 预训练权重
├── scripts/
│   ├── separate_species.py        # 按元数据分离人类和小鼠切片
│   ├── split_dataset.py           # 按完整切片划分 train/val/test
│   └── dataset_files.py           # 成对文件操作与路径规则
├── dataloader/
│   ├── HESTDataset.py              # 按切片读取图像、标签和稀疏邻接矩阵
│   └── make_adjacency.py           # 按空间距离构建带自连接的邻接矩阵
├── model/
│   ├── mymodel.py                  # UNI → GCN → ResNet
│   ├── uni.py                      # 图像特征编码器
│   ├── gcn.py                      # 空间邻居信息聚合
│   └── resnet.py                   # 基因表达预测头
├── ckpts/                         # 下载权重后生成
│   └── pytorch_model.bin           # UNI 预训练权重
├── hest_data/                     # 下载数据后生成
│   ├── patches/
│   │   ├── TENX96.h5
│   │   ├── TENX99.h5
│   │   └── ...
│   └── st/
│       ├── TENX96.h5ad
│       ├── TENX99.h5ad
│       └── ...
└── outputs/                       # 训练时生成
    └── train/
        ├── settings.json          # 本次训练参数、切片和基因顺序
        ├── loss_history.json      # 各 epoch 的训练 MSE
        ├── val_loss_history.json  # 指定 --val-root 时生成，记录验证 MSE
        ├── best.pt                # 验证 MSE 最低的 epoch；指定 --val-root 时生成
        └── last.pt                # 最近完成 epoch 的模型和优化器状态
```

`ckpts/`、`hest_data/` 和 `outputs/train/` 是权重、数据和运行产物的位置，不表示克隆仓库后已经存在这些文件。`TENX96`、`TENX99` 是同名图像与表达文件的配对示例。

## 训练输入路径

| 用途 | 项目相对路径 | 当前本地默认路径 |
|---|---|---|
| 训练入口 | `train.py` | `E:\workplace\HiST-FEAST\train.py` |
| 图像 patch 目录 | `hest_data/patches/` | `E:\workplace\HiST-FEAST\hest_data\patches` |
| 表达标签目录 | `hest_data/st/` | `E:\workplace\HiST-FEAST\hest_data\st` |
| UNI 预训练权重 | `ckpts/pytorch_model.bin` | `E:\workplace\HiST-FEAST\ckpts\pytorch_model.bin` |

图像和表达按切片文件名配对：

```text
hest_data/patches/TENX96.h5  ↔  hest_data/st/TENX96.h5ad
hest_data/patches/TENX99.h5  ↔  hest_data/st/TENX99.h5ad
```

- `.h5` 中读取 `img`、`coords`，以及 `barcode` 或 `barcodes`。
- `.h5ad` 中读取表达矩阵 `X`；按 patch barcode 对齐行，按统一的 `gene_names` 对齐列。
- 每个 Dataset 样本是一张切片：图像为 `[N, 3, 224, 224]`，稀疏邻接矩阵为 `[N, N]`，标签为 `[N, G]`。
- 训练直接读取 `patches/` 和 `st/`，当前流程不要求读取完整 WSI、单条转录本或细胞分割文件。

## 下载配置与下载路径

两个下载脚本均读取：

```text
config/token.json
```

该本地文件的格式为：

```json
{
  "hf_token": "YOUR_HUGGING_FACE_TOKEN"
}
```

`config/token.json` 已被当前 `.gitignore` 忽略。训练本身不读取这个 token；它用于下载所需数据和权重。

| 脚本 | 保存位置 |
|---|---|
| `download/HEST_dataset_download.py` | 项目根目录下的 `hest_data/` |
| `download/UNI_ckpt_download.py` | 项目根目录下的 `ckpts/pytorch_model.bin` |

HEST 下载脚本从 Hugging Face 读取最新版本的 HEST 元数据，按类别筛选全部切片。默认只获取小型元数据 CSV 和远程文件大小，显示每张切片的 patches、st 大小和总量；加上 `--download` 才下载切片文件。

```powershell
# 查看器官类别及各类别的切片数
python download/HEST_dataset_download.py --list-values organ

# 预览 Breast 全部切片的 patches + st 大小
python download/HEST_dataset_download.py --organ Breast

# 确认大小后下载
python download/HEST_dataset_download.py --organ Breast --download

# 进一步限定为人类、Xenium、IDC；多个条件取交集
python download/HEST_dataset_download.py --organ Breast --filter "species=Homo sapiens" --filter "st_technology=Xenium" --filter "oncotree_code=IDC"

# 查看其他分类，例如测序技术；也可用 --list-values species 或 oncotree_code
python download/HEST_dataset_download.py --list-values st_technology

# 可选：保存逐文件字节数清单，并指定独立的数据目录
python download/HEST_dataset_download.py --organ Breast --local-dir hest_data/Breast --report outputs/breast_download.csv
```

`--organ` 和 `--filter` 的值忽略大小写，采用完整值匹配；支持元数据中的任意列。只会下载精确匹配的 `patches/<id>.h5` 与 `st/<id>.h5ad`，不会附带 WSI、转录本或分割文件。如果任意匹配切片缺少其中一个文件，脚本会报出缺失清单并停止。

总大小按远程完整文件字节数统计，已存在的本地文件也计入；它不是剩余下载流量，也不包括缓存、元数据及传输开销。实际下载由 Hugging Face 处理已有文件和缓存。预览会显示仓库 commit；在下载命令加上 `--revision <该 commit>`，可固定为同一数据版本。可用 `--metadata-file HEST_v1_3_0.csv` 指定仓库中的元数据版本。

默认保存到 `hest_data/patches/` 和 `hest_data/st/`。如果不同器官需要分别训练，可像上例用 `--local-dir hest_data/Breast` 单独保存，并用 `python train.py --root hest_data/Breast ...` 训练；当前训练脚本会读取指定目录中的全部切片。

## 按物种分离数据

在项目根目录运行：

```bash
python scripts/separate_species.py --root hest_data --output-dir hest_data_species
```

脚本只获取官方 HEST 的小型元数据 CSV，使用 `id` 对应的 `species` 分类已有文件，不会重新下载 patches 或 st。默认从 `config/token.json` 读取 token。也可提供本地元数据 CSV，离线完成分类：

```bash
python scripts/separate_species.py --root hest_data --output-dir hest_data_species --metadata /path/to/HEST_v1_3_0.csv
```

生成结构：

```text
hest_data_species/
├── manifest.json
├── human/
│   ├── patches/*.h5
│   └── st/*.h5ad
└── mouse/
    ├── patches/*.h5
    └── st/*.h5ad
```

`Homo sapiens` 对应 `human/`，`Mus musculus` 对应 `mouse/`。混合物种及其他标注会进入额外的 `other/`，不会归入人类或小鼠。元数据中不存在的切片会报错。

## 按 7:2:1 划分训练、验证和测试集

选择一个包含 `patches/` 和 `st/` 的 dataset 文件夹，例如分离后的小鼠数据：

```bash
python scripts/split_dataset.py --root hest_data_species/mouse --output-dir hest_data_splits/mouse --seed 42
```

生成结构：

```text
hest_data_splits/mouse/
├── manifest.json
├── train/
│   ├── patches/*.h5
│   └── st/*.h5ad
├── val/
│   ├── patches/*.h5
│   └── st/*.h5ad
└── test/
    ├── patches/*.h5
    └── st/*.h5ad
```

划分单位是一张完整切片，同名 `.h5` 和 `.h5ad` 始终分在同一组，同一切片不会跨集合。默认种子为 42；输入切片清单和种子相同，划分结果相同。非 10 的整数倍采用最大余数法分配整数数量：例如 171 张分为 120/34/17，43 张分为 30/9/4。至少需要 3 张切片，小样本优先保证三组非空。

这是按切片的随机划分，没有按患者分组；同一患者可能有多张切片，若实验要求患者之间独立，需要进一步按患者划分。

两个脚本的共同规则：

- 默认 `--mode hardlink`，使用硬链接保留原始文件路径并节省磁盘空间。硬链接共享文件内容，适合只读训练；修改任一链接的文件内容会影响另一处。
- 需要独立文件副本时加 `--mode copy`；源目录与输出目录位于不同文件系统时也需要此选项。
- 加 `--dry-run` 只预览分类或划分数量，不生成 dataset 文件。
- 输入文件必须全部成对；缺少 patches 或 st 时会停止。输出目录必须不存在或为空，避免混入旧的分类或划分。
- `manifest.json` 记录每张切片的去向、原始目录、操作方式，以及元数据来源或随机种子。
- 所有相对路径均以项目根目录为基准。脚本只整理文件，不修改表达值或处理重复基因名。

使用划分后的训练集：

```bash
python train_patch.py --root hest_data_splits/mouse/train --val-root hest_data_splits/mouse/val --genes 100 --check-data
python train_patch.py --root hest_data_splits/mouse/train --val-root hest_data_splits/mouse/val --genes 100 --epochs 3 --lr 1e-4 --uni-batch-size 64
```

`--root` 指向训练集，`--val-root` 指向验证集。指定验证集后，两个训练入口均在每个 epoch 的训练完成后执行验证，打印 Train MSE 和 Val MSE，并保存验证 MSE 最低的 `best.pt`；没有指定 `--val-root` 时仍执行原来的纯训练流程。训练入口不自动执行 test 评估，独立测试使用 `evaluate.py`。

目标基因仅从训练集选择；验证集使用完全相同的名称和列顺序，不重新选基因。启动前检查两组切片 ID 是否重叠、barcode/坐标/图像形状及目标基因是否有效；验证集为空、缺少目标基因或出现重复 barcode/基因名时会报错。`--check-data` 会检查两组数据后退出，不加载 UNI、不训练。

验证使用 `model.eval()` 和 `torch.no_grad()`，不反向传播、不更新参数。`train.py` 验证时仍一次处理整张切片；`train_patch.py` 验证时按 `--uni-batch-size` 提取 UNI 特征，再执行完整切片的 GCN 和 ResNet。训练和验证的 epoch MSE 都是各切片 MSE 的算术平均，每张切片等权。重复基因名仍不会自动合并；按物种分离、划分目录也不会消除原始数据中的重复名称。

例如，使用整张切片前向的训练入口：

```bash
python train.py --root hest_data_splits/mouse/train --val-root hest_data_splits/mouse/val --genes 100 --epochs 3 --lr 1e-4
```

## 运行方式

在已安装 PyTorch、torchvision、timm、anndata、h5py、numpy、scipy、scikit-learn、Pillow 和 tqdm 的 Python 环境中，从项目根目录运行：

```powershell
python download/HEST_dataset_download.py --organ Breast
python download/HEST_dataset_download.py --organ Breast --download
python download/UNI_ckpt_download.py
python train.py --check-data
python train.py --genes 100 --epochs 3
```

`--check-data` 检查训练集及已指定的验证集的图像形状、坐标及 barcode/基因对齐，随后退出，不加载 UNI 或开始训练。

直接执行 `python train.py` 时，默认使用 10 个共同目标基因、3 个 epoch、学习率 `1e-4`、建图半径 550 原图像素，自动选择 CUDA 或 CPU。

### UNI 分批特征提取

当整张切片送入 UNI 导致显存不足时，使用 `train_patch.py`：

```powershell
python train_patch.py --genes 100 --epochs 3 --uni-batch-size 64
```

原始图像保留在 CPU，每次只将指定数量的 patch 送入冻结的 UNI，归一化也在每个小批次内部执行。特征按原 patch 顺序汇总，再用完整邻接矩阵执行 GCN 和 ResNet，每张切片更新一次参数。`--uni-batch-size` 默认为 64，可根据显存调整；其余训练参数及数据检查方式与 `train.py` 相同，默认输出目录为 `outputs/train_patch/`。

分批只降低 UNI 的前向显存峰值，GCN 和 ResNet 仍处理整张切片，CPU 仍需容纳整张切片的原始 patch。可用 `--val-root` 加入验证，验证的 UNI 前向也使用相同分批大小。

## 在测试集上评估已训练模型

先检查测试集与 checkpoint 是否匹配：

```bash
python evaluate.py --root hest_data_splits/mouse/test --checkpoint outputs/train_patch_mouse/best.pt --check-data
```

检查会直接采用 checkpoint 中保存的基因及顺序、建图半径，核对测试集基因和 barcode。当前 evaluation 不检查测试切片是否与训练/验证切片重叠，需要自行选择独立的 test 目录。不需要提供 `--genes`，也不会从测试集重新筛选目标基因。`--check-data` 不加载 UNI、不推理、不写测试结果。

通过检查后执行：

```bash
python evaluate.py \
  --root hest_data_splits/mouse/test \
  --checkpoint outputs/train_patch_mouse/best.pt \
  --output-dir outputs/test_mouse \
  --uni-batch-size 64 \
  --device auto
```

该脚本可读取 `train.py` 或 `train_patch.py` 保存的 checkpoint，严格加载 GCN 和 ResNet 参数；需要项目中的 `ckpts/pytorch_model.bin` 与训练时的 UNI 权重相同，因为训练 checkpoint 没有重复保存冻结的 UNI。推理使用 eval 和 no_grad，UNI 按 patch 分批，GCN 和 ResNet 处理完整切片。

`--uni-batch-size` 可省略，默认读取训练设置中的大小，没有记录时使用 64。所有路径均相对于项目根目录；`--checkpoint` 也可以指向下载回本地的 `server_results/outputs/train_patch_mouse/best.pt`，但本地评估还需要完整 test 数据、UNI 权重和运行依赖。

输出文件：

| 文件 | 内容 |
|---|---|
| `metrics.json` | checkpoint、基因顺序、测试切片，以及 HiST 口径的总体/逐切片 R² 和 PCC |
| `slide_metrics.csv` | 每张切片的 patch 数、R2、PCC |
| `gene_metrics.csv` | 每个基因的 R2、PCC，作为补充分析 |
| `predictions/<id>.npz` | 仅加 `--save-predictions` 时生成：prediction、target、gene_names、barcodes，全部按 patch 顺序对齐 |

评估指标采用 [HiST 官方代码](https://github.com/wwyi1828/HiST/blob/main/tasks/gene_prediction/trainer.py) 的两种汇总方式，计算定义对应其 [`r2_score` 和 `pearsonr`](https://github.com/wwyi1828/HiST/blob/main/lib/utils/metrics.py)：

- `spot_metrics.R2` / `spot_metrics.PCC`：与 HiST 一样，使用 `np.vstack` 汇总测试切片，将全部 spot × gene 表达值展平成一维，然后直接调用 `sklearn.metrics.r2_score(true, pred)` 和 `scipy.stats.pearsonr(true, pred)[0]`。预测和真值以 float64 保存在 CPU 内存中。
- `slide_avg_metrics.R2` / `slide_avg_metrics.PCC`：每张切片先展平并独立计算 R²、PCC，再对切片取等权平均。
- `slide_std_metrics`：逐切片指标的总体标准差（`ddof=0`），与官方评估函数一致。它是切片间标准差，不是论文中四折交叉验证的折间标准差；本脚本评估单个 checkpoint 和指定 test 集。
- `gene_metrics.csv`：另按每个基因在全部测试 patch 上直接调用同一组 sklearn/scipy 函数，便于定位表现较好的基因；该文件的基因平均不能替代上面的 HiST 总体指标。

R² 使用 `1 - SSE/SST`，不是 PCC 的平方，负值会原样保留。真实值为常数时遵循 sklearn 的默认规则：完全预测正确时 R²=1，否则为 0。PCC 在预测或真值为常数时未定义；指标向量少于 2 个元素时两项均未定义。未定义值在 JSON 中用 null、CSV 中留空；逐切片均值/标准差中若存在未定义值，则相应汇总也为 null，保持官方 `np.mean/np.std` 的传播语义，不静默丢弃切片。`valid_slide_metrics` 和 `valid_gene_metrics` 给出有效数量。

指标在线性表达空间计算，与 HiST 官方最终评估的尺度一致。当前项目模型已直接输出原始计数，因此不额外应用 `log1p` 或 `expm1`。本次只调整测试指标，训练仍使用原有原始计数 MSE 和最低 Val MSE 选择 `best.pt`。

例如，导出预测和真值以便后续分析：

```bash
python evaluate.py --root hest_data_splits/mouse/test --checkpoint outputs/train_patch_mouse/best.pt --output-dir outputs/test_mouse --save-predictions
```

## 路径定位规则

1. `train.py` 通过 `Path(__file__).resolve().parent` 定位项目根目录。
2. `--root`、`--val-root` 和 `--output-dir` 的相对路径均以项目根目录为基准；绝对路径直接使用。
3. `--root` 指向同时包含 `patches/` 和 `st/` 的数据目录，而不是其中某一个子目录。
4. UNI 权重由 `model/mymodel.py` 定位到项目根目录下的 `ckpts/pytorch_model.bin`。
5. 两个下载脚本同样按自身文件位置定位项目根目录并读取 `config/token.json`；HEST 默认写入 `hest_data/`，可通过 `--local-dir` 修改，UNI 写入 `ckpts/`。
6. 当前 `train.py` 通过命令行接收训练参数，不读取 `config/train.json`。

因此，从其他工作目录启动 `train.py` 时，默认数据和输出位置仍然是这个项目下的目录。Notebook 中自行使用的 `./` 路径则以 Notebook 内核的当前工作目录为准。

例如，在项目根目录将数据和输出改到其他绝对路径：

```powershell
python train.py --root "E:\datasets\hest_data" --output-dir "E:\experiments\hist_run1" --genes 100 --epochs 3
```

## 训练输出路径

默认输出目录：

```text
E:\workplace\HiST-FEAST\outputs\train
```

| 输出文件 | 内容 |
|---|---|
| `settings.json` | 命令行参数、解析后的路径、训练及验证切片、目标基因及列顺序 |
| `loss_history.json` | 每个 epoch 的平均训练 MSE，统计方式为各切片 MSE 的平均值 |
| `val_loss_history.json` | 指定验证集时生成；每个 epoch 的平均验证 MSE，统计方式同训练 |
| `last.pt` | 最近完成 epoch 的 GCN、ResNet、优化器状态、基因列表、训练设置、训练/验证历史及最佳验证指标 |
| `best.pt` | 指定验证集时生成；最低 Val MSE 对应 epoch 的 checkpoint，误差相同时保留较早的 epoch |

这些文件在训练过程中写入和更新；建议不同实验使用不同的 `--output-dir`。冻结的 UNI 权重不重复写入 checkpoint；重新构建模型仍需要原始 `ckpts/pytorch_model.bin`。当前脚本没有 `--resume` 自动恢复入口。

## 当前训练行为

- 直接复用 `dataloader/HESTDataset.py` 和原来的 `MyModel.forward()`。
- 一张切片对应一个训练样本和一次优化器更新，`batch_size=1`、`num_workers=0`。
- 每次将整张切片的全部 patch、邻接矩阵和标签送到训练设备，直接调用 `model(patches, adjacency)`。
- 图像通过 `PILToTensor()` 保持 uint8，由 UNI 内部执行图像归一化；没有 patch 分批计算或激活 checkpoint。
- UNI 冻结，训练 GCN 和 ResNet；邻接矩阵保留未归一化形式及自连接。
- 标签使用原始存储计数，损失为 MSE；当前没有 log1p 或总量归一化。
- `--root` 中的配对切片参与训练；可用 `--val-root` 指定单独的验证集。划分目录使用 `scripts/split_dataset.py`，训练入口不重新划分数据、不自动读取 test。
- 目标基因取所有切片的共有真实靶基因，排除 BLANK 和阴性对照，按第一张切片的顺序选前 G 个；当前没有 HVG 排名。

