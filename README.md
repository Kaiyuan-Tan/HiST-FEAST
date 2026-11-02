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
├── GCN.ipynb                       # 实验 Notebook
├── .gitignore                      # 忽略本地 token 配置
├── config/
│   ├── token.json                  # 下载脚本使用的 Hugging Face token
│   └── train.json                  # 现有配置文件；当前 train.py 不读取它
├── download/
│   ├── HEST_dataset_download.py    # 下载 HEST 图像 patch 和表达数据
│   └── UNI_ckpt_download.py        # 下载 UNI 预训练权重
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
- 同名基因在共同基因统计中只计一次；读取标签时，按 spot 将同名列的原始计数相加，并在相加前转为 float32。barcode 重复仍会报错。原始 `.h5ad` 文件不会被修改，也不会给基因名自动添加后缀。
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

## 运行方式

在已安装 PyTorch、torchvision、timm、anndata、h5py、numpy、scipy、scikit-learn、Pillow 和 tqdm 的 Python 环境中，从项目根目录运行：

```powershell
python download/HEST_dataset_download.py --organ Breast
python download/HEST_dataset_download.py --organ Breast --download
python download/UNI_ckpt_download.py
python train.py --check-data
python train.py --genes 100 --epochs 3
```

`--check-data` 检查图像形状、坐标及 barcode/基因对齐，随后退出，不加载 UNI 或开始训练。

直接执行 `python train.py` 时，默认使用 10 个共同目标基因、3 个 epoch、学习率 `1e-4`、建图半径 550 原图像素，自动选择 CUDA 或 CPU。

### UNI 分批特征提取

当整张切片送入 UNI 导致显存不足时，使用 `train_patch.py`：

```powershell
python train_patch.py --genes 100 --epochs 3 --uni-batch-size 64
```

原始图像保留在 CPU，每次只将指定数量的 patch 送入冻结的 UNI，归一化也在每个小批次内部执行。特征按原 patch 顺序汇总，再用完整邻接矩阵执行 GCN 和 ResNet，每张切片更新一次参数。`--uni-batch-size` 默认为 64，可根据显存调整；其余训练参数及数据检查方式与 `train.py` 相同，默认输出目录为 `outputs/train_patch/`。

分批只降低 UNI 的前向显存峰值，GCN 和 ResNet 仍处理整张切片，CPU 仍需容纳整张切片的原始 patch。

## 路径定位规则

1. `train.py` 通过 `Path(__file__).resolve().parent` 定位项目根目录。
2. `--root` 和 `--output-dir` 的相对路径均以项目根目录为基准；绝对路径直接使用。
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
| `settings.json` | 命令行参数、解析后的路径、参与训练的切片、目标基因及列顺序 |
| `loss_history.json` | 每个 epoch 的平均训练 MSE，统计方式为各切片 MSE 的平均值 |
| `last.pt` | epoch、GCN 参数、ResNet 参数、优化器状态、基因列表、训练设置与 loss 历史 |

这些文件在训练过程中写入和更新。冻结的 UNI 权重不重复写入 `last.pt`；重新构建模型仍需要原始 `ckpts/pytorch_model.bin`。当前脚本没有 `--resume` 自动恢复入口。

## 当前训练行为

- 直接复用 `dataloader/HESTDataset.py` 和原来的 `MyModel.forward()`。
- 一张切片对应一个训练样本和一次优化器更新，`batch_size=1`、`num_workers=0`。
- 每次将整张切片的全部 patch、邻接矩阵和标签送到训练设备，直接调用 `model(patches, adjacency)`。
- 图像通过 `PILToTensor()` 保持 uint8，由 UNI 内部执行图像归一化；没有 patch 分批计算或激活 checkpoint。
- UNI 冻结，训练 GCN 和 ResNet；邻接矩阵保留未归一化形式及自连接。
- 标签使用原始存储计数，损失为 MSE；当前没有 log1p 或总量归一化。
- 所有配对切片都参与训练；当前没有自动划分验证集或测试集。
- 目标基因取所有切片的共有真实靶基因，排除 BLANK 和阴性对照，按第一张切片的顺序选前 G 个；当前没有 HVG 排名。

