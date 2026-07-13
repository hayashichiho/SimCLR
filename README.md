# SimCRL

STL-10を用いて、自己教師あり表現学習手法 **SimCLR（Simple Framework for Contrastive Learning of Visual Representations）** を実装・評価するJupyter Notebookです。

画像のクラスラベルを使用せずにエンコーダを事前学習し、得られた特徴表現が画像分類にどの程度有効かを、線形評価によって確認します。

## 概要

通常の教師あり学習では、画像とそのクラスラベルを用いてモデルを学習します。一方、SimCLRでは、同じ画像から生成した2種類の拡張画像を正例ペアとして扱い、ラベルを使用せずに画像の特徴表現を学習します。

本ノートブックでは、STL-10のラベルなし画像を用いてSimCLRの事前学習を行った後、エンコーダを固定した状態で線形分類器のみを学習します。

主に、以下の4つの特徴表現を比較します。

| 評価条件         | 特徴抽出器                   | 評価する特徴                |
| ------------ | ----------------------- | --------------------- |
| Scratch `h`  | ランダム初期化ResNet-18        | エンコーダ出力 `h`           |
| ImageNet `h` | ImageNet事前学習済みResNet-18 | エンコーダ出力 `h`           |
| SimCLR `h`   | SimCLR事前学習済みResNet-18   | エンコーダ出力 `h`           |
| SimCLR `z`   | SimCLR事前学習済みモデル         | Projection Head出力 `z` |

これにより、SimCLRによって獲得された特徴表現の有効性と、エンコーダ出力 `h` とProjection Head出力 `z` の違いを確認します。

## 実験の流れ

```text
STL-10のラベルなし画像
        │
        ├── データ拡張 t1 ──> x_i
        │
        └── データ拡張 t2 ──> x_j
                    │
                    ▼
            ResNet-18 Encoder
                    │
                    ▼
           特徴表現 h（512次元）
                    │
                    ▼
            Projection Head
                    │
                    ▼
           投影表現 z（128次元）
                    │
                    ▼
              NT-Xent Loss
```

実験は大きく以下の3段階で構成されています。

1. STL-10のラベルなし画像を用いたSimCLR事前学習
2. PCAによる特徴表現 `h` と `z` の可視化
3. エンコーダを固定した線形評価

## 主な実装内容

* STL-10のラベルなしデータを用いた自己教師あり学習
* 同一画像から2つの拡張画像を生成する`TwoCropsTransform`
* ResNet-18を用いた特徴抽出
* 2層MLPによるProjection Head
* NT-Xent損失による対照学習
* LARSを用いた最適化
* Linear WarmupとCosine Annealingによる学習率制御
* チェックポイントの保存と学習再開
* PCAによる特徴空間の可視化
* 凍結した特徴抽出器に対する線形評価
* Scratch、ImageNet、SimCLRの比較

## 使用データセット

### STL-10

[STL-10](https://cs.stanford.edu/~acoates/stl10/)は、10クラスの自然画像から構成される画像認識用データセットです。

各画像のサイズは`96 × 96`ピクセルです。

本実験では、以下のデータを使用します。

| 用途         |       split |      画像数 | ラベルの使用 |
| ---------- | ----------: | -------: | ------ |
| SimCLR事前学習 | `unlabeled` | 100,000枚 | 使用しない  |
| 線形分類器の学習   |     `train` |   5,000枚 | 使用する   |
| 線形評価       |      `test` |   8,000枚 | 使用する   |

データは初回実行時に`./data`へ自動的にダウンロードされます。

## データ拡張

SimCLRでは、同じ画像に異なるデータ拡張を適用し、生成された2枚の画像を正例ペアとして扱います。

本実装では、以下のデータ拡張を使用します。

| データ拡張                  | 設定                             | 目的                    |
| ---------------------- | ------------------------------ | --------------------- |
| Random Resized Crop    | `scale=(0.08, 1.0)`            | 位置やスケールの変化に頑健な表現を学習する |
| Random Horizontal Flip | 確率0.5                          | 左右反転に対する不変性を獲得する      |
| Color Jitter           | 強度`0.4, 0.4, 0.4, 0.1`、適用確率0.8 | 色や明るさへの依存を抑える         |
| Random Grayscale       | 適用確率0.2                        | 色情報だけに依存しない特徴を学習する    |
| Gaussian Blur          | カーネルサイズは画像サイズから算出              | 細かなテクスチャ変化に対して頑健にする   |
| ToTensor               | ―                              | PyTorch Tensorへ変換する   |

評価時にはランダムなデータ拡張を行わず、以下の処理のみを適用します。

```text
Resize
  ↓
Center Crop
  ↓
ToTensor
```

## モデル構成

### Encoder

特徴抽出器には`torchvision.models.resnet18`を使用します。

STL-10の画像サイズに合わせるため、通常のResNet-18から以下の変更を行っています。

* 最初の畳み込み層を`7 × 7、stride=2`から`3 × 3、stride=1`へ変更
* 最初のMax Pooling層を削除
* 最終全結合層を削除
* エンコーダ出力を512次元の特徴ベクトル`h`として使用

```text
Input image: 3 × 96 × 96
        │
        ▼
Modified ResNet-18
        │
        ▼
Feature representation h: 512 dimensions
```

### Projection Head

SimCLRの事前学習では、エンコーダ出力`h`を直接損失計算に使用せず、Projection Headによって低次元空間へ変換します。

本実装のProjection Headは以下の構成です。

```text
Linear: 512 → 512
Batch Normalization
ReLU
Linear: 512 → 128
Batch Normalization
```

Projection Headの出力`z`は128次元です。

```text
h: 512 dimensions
        │
        ▼
Projection Head
        │
        ▼
z: 128 dimensions
```

## NT-Xent Loss

学習には、SimCLRで提案された **Normalized Temperature-scaled Cross Entropy Loss（NT-Xent Loss）** を使用します。

同じ元画像から生成された2枚の拡張画像を正例、それ以外の画像を負例として扱います。

正例ペア`(i, j)`に対する損失は、概念的には次式で表されます。

[
\ell_{i,j}
==========

-\log
\frac{
\exp(\operatorname{sim}(z_i,z_j)/\tau)
}{
\sum_{k \neq i}
\exp(\operatorname{sim}(z_i,z_k)/\tau)
}
]

ここで、

* (z_i, z_j)：Projection Headから得られた正規化済み特徴ベクトル
* (\operatorname{sim}(\cdot,\cdot))：コサイン類似度
* (\tau)：Temperatureパラメータ

です。

本実装では、Temperatureを`0.1`に設定しています。

損失を最小化することで、以下のような特徴空間を学習します。

* 同じ画像から生成された2つの特徴を近づける
* 異なる画像から生成された特徴を遠ざける

## デフォルトの実験設定

| パラメータ               |          設定値 |
| ------------------- | -----------: |
| 入力画像サイズ             |    `96 × 96` |
| バッチサイズ              |        `256` |
| 事前学習エポック数           |        `300` |
| Warmupエポック数         |         `10` |
| 基準学習率               |        `0.5` |
| Weight Decay        | `1.0 × 10⁻⁴` |
| Temperature         |        `0.1` |
| エンコーダ出力次元           |        `512` |
| Projection Head出力次元 |        `128` |
| DataLoader workers  |          `2` |
| Random Seed         |         `42` |

実際の学習率は、バッチサイズに応じて次のように設定されます。

[
\mathrm{learning\ rate}
=======================

\mathrm{base\ learning\ rate}
\times
\frac{\mathrm{batch\ size}}{256}
]

デフォルト設定では、バッチサイズが256であるため、学習率は`0.5`です。

## 最適化手法

SimCLRの事前学習では、以下の設定を使用します。

* Optimizer：SGDを基にしたLARS
* Momentum：`0.9`
* Weight Decay：`1.0 × 10⁻⁴`
* Learning Rate Scheduler：Linear Warmup + Cosine Annealing

最初の10エポックでは学習率を線形に増加させ、その後はCosine Annealingによって徐々に減少させます。

BiasおよびBatch Normalizationのパラメータには、LARSによるLayer-wise adaptationを適用しません。

## 必要な環境

CUDAに対応したGPU環境での実行を推奨します。

CPUのみでも実行できますが、300エポックの事前学習には長い時間がかかります。

主な使用ライブラリは以下のとおりです。

* Python
* PyTorch
* torchvision
* NumPy
* matplotlib
* Pillow
* tqdm
* scikit-learn
* JupyterLabまたはJupyter Notebook

## セットアップ

### 1. リポジトリをクローン

```bash
git clone https://github.com/hayashichiho/SimCRL.git
cd SimCRL
```

### 2. 仮想環境を作成

macOSまたはLinuxの場合：

```bash
python -m venv .venv
source .venv/bin/activate
```

Windowsの場合：

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. 必要なライブラリをインストール

```bash
pip install torch torchvision numpy matplotlib pillow tqdm scikit-learn pandas jupyterlab
```

CUDAを使用する場合は、使用するCUDAバージョンに対応したPyTorchをインストールしてください。

### 4. JupyterLabを起動

```bash
jupyter lab
```

起動後、`SimCRL.ipynb`を開きます。

## 実行方法

Notebookのセルを上から順番に実行してください。

処理は主に以下の順序で進みます。

### 1. ライブラリのインポートと設定

使用するデバイス、バッチサイズ、エポック数、学習率などを設定します。

CUDAを利用できる場合はGPU、利用できない場合はCPUが自動的に選択されます。

### 2. STL-10のダウンロード

STL-10の`unlabeled`、`train`、`test` splitを`./data`へダウンロードします。

### 3. SimCLR事前学習

STL-10のラベルなし画像を用いて、ResNet-18 EncoderとProjection Headを学習します。

事前学習にはデフォルトで300エポック必要です。

### 4. 特徴表現の抽出

事前学習済みモデルを読み込み、ラベル付きのSTL-10画像から以下の特徴を抽出します。

* エンコーダ出力`h`：512次元
* Projection Head出力`z`：128次元

抽出される特徴量の形状は以下のとおりです。

| データ          |        `h`の形状 |        `z`の形状 |
| ------------ | ------------: | ------------: |
| STL-10 train | `(5000, 512)` | `(5000, 128)` |
| STL-10 test  | `(8000, 512)` | `(8000, 128)` |

### 5. PCAによる特徴空間の可視化

`h`および`z`を標準化した後、主成分分析によって2次元へ圧縮します。

クラスごとに色分けして描画することで、各特徴空間におけるクラスの分離状態を確認します。

### 6. 線形評価

特徴抽出器を固定し、その出力に対する線形分類器のみを学習します。

線形評価では、エンコーダのパラメータは更新されません。

## 線形評価の設定

線形分類器は以下の構成です。

```text
Input feature
      │
      ▼
Dropout
      │
      ▼
Linear layer
      │
      ▼
10-class prediction
```

学習条件は以下のとおりです。

| パラメータ        |               設定値 |
| ------------ | ----------------: |
| エポック数        |             `100` |
| Optimizer    |             AdamW |
| 学習率          |      `1.0 × 10⁻³` |
| Weight Decay |      `1.0 × 10⁻⁴` |
| Scheduler    | CosineAnnealingLR |
| Dropout      |             `0.5` |
| 出力クラス数       |              `10` |

学習中は、以下の値を記録します。

* Training Loss
* Training Accuracy
* Test Accuracy
* Best Test Accuracy

## 各評価条件の意味

### Scratch `h`

ランダムに初期化したResNet-18の出力を使用します。

エンコーダを学習せずに固定するため、画像分類に有効な特徴表現を持っていない場合の基準になります。

### ImageNet `h`

ImageNetで教師あり事前学習されたResNet-18の出力を使用します。

大規模なラベル付き画像データによって獲得された一般的な画像特徴と、SimCLRによる自己教師あり特徴を比較します。

### SimCLR `h`

SimCLRで事前学習したResNet-18 Encoderの出力を使用します。

通常、下流タスクではProjection Headを取り除き、この`h`を特徴表現として使用します。

### SimCLR `z`

SimCLRのProjection Head出力を使用します。

`z`は対照学習の損失を計算するために最適化された表現であり、下流タスクに適した`h`とは異なる性質を持つ可能性があります。

`h`と`z`の評価結果を比較することで、Projection Headが特徴表現に与える影響を確認できます。

## 実行結果の例

Notebookに保存されている実行例では、SimCLR事前学習の損失は以下のように減少しています。

| エポック | Training Loss |
| ---: | ------------: |
|    1 |     約`6.0964` |
|  300 |     約`3.8310` |

学習の進行に伴ってNT-Xent Lossが減少しており、正例ペアと負例を区別する特徴表現が学習されていることを確認できます。

実際の損失や分類精度は、以下の条件によって変化する可能性があります。

* GPUやPyTorchの実行環境
* CUDAおよびcuDNNのバージョン
* データ拡張の乱数
* バッチサイズ
* 学習率
* 学習エポック数

線形評価では、次の形式で結果を比較できます。

| 評価条件         | 特徴次元 | Best Test Accuracy |
| ------------ | ---: | -----------------: |
| Scratch `h`  |  512 |   Notebookの実行結果を参照 |
| ImageNet `h` |  512 |   Notebookの実行結果を参照 |
| SimCLR `h`   |  512 |   Notebookの実行結果を参照 |
| SimCLR `z`   |  128 |   Notebookの実行結果を参照 |

## 出力ファイル

学習結果は`simclr_output`ディレクトリへ保存されます。

```text
simclr_output/
├── checkpoint.pth
├── pretrained_simclr.pth
└── loss_curve.png
```

### `checkpoint.pth`

学習途中の状態を保存したチェックポイントです。

以下のような情報を含み、学習を途中から再開するために使用されます。

* 現在のエポック
* モデルのパラメータ
* Optimizerの状態
* Learning Rate Schedulerの状態
* 学習履歴

チェックポイントは10エポックごとに更新されます。

### `pretrained_simclr.pth`

300エポックの事前学習が完了したSimCLRモデルのパラメータです。

特徴抽出、PCAによる可視化、線形評価に使用します。

### `loss_curve.png`

SimCLR事前学習時のTraining Lossをエポックごとに可視化した画像です。

## 学習の再開

`simclr_output/checkpoint.pth`が存在する場合は、保存されたエポックから学習を再開します。

既に`pretrained_simclr.pth`が存在する場合は、事前学習済みモデルを利用できます。

学習を最初からやり直す場合は、既存のチェックポイントを別の場所へ移動するか、削除してください。

## ディレクトリ構成

```text
SimCRL/
├── README.md
├── SimCRL.ipynb
├── data/
│   └── stl10_binary/
└── simclr_output/
    ├── checkpoint.pth
    ├── pretrained_simclr.pth
    └── loss_curve.png
```

`data`および`simclr_output`は、Notebookの実行によって生成されます。

## 再現性について

乱数シードには`42`を設定しています。

```python
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
```

CUDAを使用する場合にも乱数シードを設定していますが、GPU上の一部の演算は非決定的に実行される可能性があります。そのため、環境によって結果が完全には一致しない場合があります。

## バッチサイズを変更する場合

GPUメモリが不足する場合は、`batch_size`を小さくしてください。

```python
batch_size = 128
```

本実装では、バッチサイズに合わせて学習率が自動的に調整されます。

ただし、SimCLRでは同一バッチ内のほかの画像を負例として利用するため、バッチサイズを小さくすると負例数も減少します。その結果、学習性能が変化する可能性があります。

## トラブルシューティング

### CUDA out of memoryが発生する

バッチサイズを小さくしてください。

```python
batch_size = 128
```

それでも解消しない場合は、`64`や`32`まで下げてください。

### `pretrained_simclr.pth`が見つからない

SimCLRの事前学習セルを先に実行してください。

モデルの保存先は以下です。

```text
./simclr_output/pretrained_simclr.pth
```

### STL-10をダウンロードできない

インターネット接続を確認し、データセットを読み込むセルを再実行してください。

既に不完全なファイルが作成されている場合は、`data`ディレクトリを削除してから再実行してください。

### CPUで学習が進まない

300エポックのSimCLR事前学習は計算量が大きいため、CUDAに対応したGPUの利用を推奨します。

動作確認のみを行う場合は、以下のようにエポック数を減らしてください。

```python
max_epochs = 10
```

ただし、エポック数を減らした場合は十分な特徴表現を学習できない可能性があります。

## 本実装の位置付け

本リポジトリは、SimCLRの基本的な仕組みを学習し、以下の処理を一つのNotebook上で確認することを目的としています。

* データ拡張による正例ペアの生成
* 対照学習による自己教師あり事前学習
* エンコーダとProjection Headの役割
* PCAによる特徴空間の確認
* 線形評価による表現性能の比較

原論文のすべての実験条件を完全に再現するものではありません。原論文では、より大規模なデータセット、モデル、バッチサイズ、計算資源が使用されています。

## 参考文献

### SimCLR

Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton,
“A Simple Framework for Contrastive Learning of Visual Representations,”
Proceedings of the 37th International Conference on Machine Learning, 2020.

* Paper: https://arxiv.org/abs/2002.05709
* Official implementation: https://github.com/google-research/simclr

### STL-10

Adam Coates, Andrew Ng, Honglak Lee,
“An Analysis of Single-Layer Networks in Unsupervised Feature Learning,”
Proceedings of the 14th International Conference on Artificial Intelligence and Statistics, 2011.

* Dataset: https://cs.stanford.edu/~acoates/stl10/
