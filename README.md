# Data Drift Detection System
**資料漂移偵測系統**

用於偵測機器學習模型輸入資料的統計分布變化，協助監控模型健康狀態、提早發現資料品質問題。

---

## 目錄

- [快速開始](#快速開始)
- [安裝](#安裝)
- [使用方式](#使用方式)
  - [基本用法](#基本用法)
  - [進階設定](#進階設定)
  - [視覺化](#視覺化)
  - [配置系統](#配置系統)
- [偵測方法](#偵測方法)
- [結果結構](#結果結構)
- [閾值與判斷準則](#閾值與判斷準則)
- [目錄結構](#目錄結構)

---

## 快速開始

```python
import pandas as pd
from data_drift_module import DataDriftDetector

ref = pd.read_csv('data/training_data.csv')
curr = pd.read_csv('data/production_data.csv')

detector = DataDriftDetector(ref)
results = detector.detect_drift(curr)

print(f"漂移特徵數: {results['summary']['drifted_features']} / {results['summary']['total_features']}")
print(detector.generate_report())
```

---

## 安裝

**環境需求**：Python >= 3.9

```bash
git clone <repo-url>
cd data_drift_module
pip install -e .
```

**相依套件**（自動安裝）：`pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`

---

## 使用方式

### 基本用法

```python
import pandas as pd
from data_drift_module import DataDriftDetector

# 載入資料
reference_data = pd.read_csv('data/training_data.csv')
current_data = pd.read_csv('data/production_data.csv')

# 建立偵測器
detector = DataDriftDetector(
    reference_data=reference_data,
    threshold=0.05,        # KS / Chi-Square p-value 閾值
    psi_threshold=0.2,     # PSI 漂移閾值
    jsd_threshold=0.1,     # JSD 漂移閾值
    min_sample_size=30     # 最小樣本數警告
)

# 執行偵測
results = detector.detect_drift(current_data)

# 查看摘要
summary = results['summary']
print(f"總特徵數: {summary['total_features']}")
print(f"漂移特徵數: {summary['drifted_features']}")
print(f"漂移比例: {summary['drift_percentage']:.1f}%")

# 取得漂移特徵列表
drifted_features = detector.get_drifted_features()

# 生成報告
print(detector.generate_report('text'))      # 文字格式
print(detector.generate_report('markdown'))  # Markdown 格式
```

### 進階設定

**啟用可選偵測方法：**

```python
detector = DataDriftDetector(
    reference_data=reference_data,
    enable_mann_whitney=True,       # 啟用 Mann-Whitney U Test
    enable_anderson_darling=True    # 啟用 Anderson-Darling Test
)
```

**使用場景配置：**

```python
from data_drift_module.config import ConfigSelector

config = ConfigSelector.select_by_scenario('production')  # 生產環境
# 可選: 'development', 'production', 'monitoring', 'research', 'strict', 'loose'

detector = DataDriftDetector(
    reference_data=reference_data,
    threshold=config['significance_level'],
    psi_threshold=config['psi_threshold'],
    jsd_threshold=config['jsd_threshold'],
    min_sample_size=config['min_sample_size']
)
```

**分析單一特徵結果：**

```python
results = detector.detect_drift(current_data)

for feature, result in results['feature_results'].items():
    if result['has_drift']:
        print(f"{feature}: {result['drift_reasons']}")

        # 數值型特徵
        if result['feature_type'] == 'numerical':
            psi = result['tests']['psi']['value']
            jsd = result['tests']['jensen_shannon_divergence']['value']
            ks_p = result['tests']['ks_test']['p_value']
            wass = result['tests']['wasserstein_distance']['value']  # 輔助指標
            print(f"  PSI={psi:.4f}, JSD={jsd:.4f}, KS p={ks_p:.4f}")

        # 類別型特徵
        elif result['feature_type'] == 'categorical':
            new_cats = result['statistics']['new_categories']
            disappeared_cats = result['statistics']['disappeared_categories']
            if new_cats:
                print(f"  新出現類別: {new_cats}")
```

### 視覺化

```python
from data_drift_module import DriftVisualizer

# 必須同時傳入 detector 和 current_data
viz = DriftVisualizer(detector, current_data)

# 漂移摘要總覽
viz.plot_drift_summary()

# 多指標比較儀表板（PSI、漂移原因、類型分析、缺失值）
viz.plot_metrics_comparison()

# 個別特徵詳細分析
viz.plot_numerical_drift('feature_name')     # 直方圖、箱型圖、Q-Q、CDF
viz.plot_categorical_drift('feature_name')   # 條形圖、差異圖、圓餅圖

# 儲存所有圖表
viz.plot_all_drifts(save_dir='drift_visualizations/')
```

### 配置系統

```python
from data_drift_module.config import DriftDetectionConfig

# 使用預設配置
config = DriftDetectionConfig.get_default_config()

# 其他配置方式
config = DriftDetectionConfig.get_strict_config()      # 嚴格（更敏感）
config = DriftDetectionConfig.get_loose_config()       # 寬鬆（較不敏感）
config = DriftDetectionConfig.get_production_config()  # 生產環境推薦

# 查看配置內容
DriftDetectionConfig.print_config(config)
```

---

## 偵測方法

### 數值型特徵

| 方法 | 角色 | 說明 |
|------|------|------|
| **KS Test** | 主力 | Kolmogorov-Smirnov 分布比較，非參數，返回 p-value |
| **PSI** | 主力 | Population Stability Index，業界金融標準，量化漂移程度 |
| **JSD** | 主力 | Jensen-Shannon Divergence，有界 [0,1]，與 KS 互補 |
| Wasserstein Distance | 輔助 | 只記錄於結果，不觸發漂移判定 |
| Mann-Whitney U | 可選 | `enable_mann_whitney=True` 啟用，非參數中位數檢定 |
| Anderson-Darling | 可選 | `enable_anderson_darling=True` 啟用 |

### 類別型特徵

| 方法 | 角色 | 說明 |
|------|------|------|
| **Chi-Square Test** | 主力 | 類別分布獨立性檢定，返回 p-value |
| **PSI（類別版）** | 主力 | 類別比例偏移量化 |
| **新/消失類別偵測** | 主力 | 直接比較類別集合，語義最清晰 |

### 附加分析（每個特徵自動執行）

| 分析 | 說明 |
|------|------|
| 缺失值漂移 | Z-test for proportions，缺失率差異 > 5% 且顯著時觸發 |
| 樣本大小驗證 | 樣本數不足時發出警告，不影響偵測結果 |

---

## 結果結構

```python
results = {
    'summary': {
        'total_features': int,       # 總特徵數
        'drifted_features': int,     # 漂移特徵數
        'drift_percentage': float,   # 漂移比例 (%)
        'warnings': list             # 欄位缺失等警告
    },
    'feature_results': {
        'feature_name': {
            'feature_type': 'numerical' | 'categorical',
            'has_drift': bool,
            'drift_reasons': list,   # 觸發漂移的指標，例如 ['ks_test', 'psi', 'jsd']
            'tests': {
                # 數值型（主力）
                'ks_test':                    {'statistic': float, 'p_value': float, 'interpretation': str},
                'psi':                        {'value': float, 'interpretation': str},
                'jensen_shannon_divergence':  {'value': float, 'interpretation': str},
                'wasserstein_distance':       {'value': float, 'interpretation': str},  # 輔助
                # 數值型（可選，啟用時才出現）
                'mann_whitney':               {'statistic': float, 'p_value': float, 'interpretation': str},
                'anderson_darling':           {'statistic': float, 'interpretation': str},
                # 類別型
                'chi_square_test':            {'statistic': float, 'p_value': float, 'interpretation': str},
                'psi':                        {'value': float, 'interpretation': str},
            },
            'statistics': {
                # 數值型: reference/current 各含 mean, std, median, min, max, q25, q75, skewness, kurtosis
                # 類別型: reference_categories, current_categories, new_categories, disappeared_categories, ...
            },
            'missing_analysis': {
                'reference_missing_rate': float,
                'current_missing_rate': float,
                'difference': float,
                'p_value': float,
                'has_drift': bool
            },
            'sample_check': {
                'is_valid': bool,
                'ref_size': int,
                'curr_size': int,
                'warnings': list
            }
        }
    }
}
```

---

## 閾值與判斷準則

### PSI 解讀

| PSI 值 | 解讀 | 建議行動 |
|--------|------|---------|
| < 0.1 | 無顯著變化 | 繼續監控 |
| 0.1 – 0.2 | 輕微變化 | 調查原因 |
| > 0.2 | 顯著漂移 | 考慮重新訓練 |

### 漂移比例對應決策

| 漂移比例 | 嚴重程度 | 建議行動 |
|---------|---------|---------|
| < 5% | 穩定 | 正常監控週期 |
| 5 – 15% | 輕微漂移 | 持續監控、記錄趨勢 |
| 15 – 30% | 中度漂移 | 密切監控、分析主要特徵、準備重新訓練 |
| ≥ 30% | 嚴重漂移 | 立即行動：檢查資料管道、評估模型效能、重新訓練 |

### 多指標一致性

- **多數主力指標同時觸發** → 高信心漂移，建議立即處理
- **單一指標觸發** → 低信心，建議進一步調查
- **樣本數不足警告** → 結果可靠性降低，建議增加樣本

---

## 目錄結構

```
data_drift_module/
├── pyproject.toml                    # 套件定義
├── README.md
├── CLAUDE.md                         # AI 協作說明文件
│
├── data_drift_module/                # Python 套件
│   ├── __init__.py                   # 公共 API
│   ├── detector.py                   # DataDriftDetector
│   ├── visualizer.py                 # DriftVisualizer
│   └── config.py                     # DriftDetectionConfig, ConfigSelector
│
└── examples/
    ├── quick_start.py                # 完整快速入門腳本
    └── notebooks/
        ├── quick_start_drift.ipynb   # 有漂移資料範例
        └── quick_start_nondrift.ipynb # 無漂移資料範例
```

**資料目錄**（不含於版本控制）：

```
data/
├── training_data.csv                 # 參考/基準資料
├── production_data_with_drift.csv    # 有漂移的生產資料
└── production_data_no_drift.csv      # 無漂移的生產資料
```
