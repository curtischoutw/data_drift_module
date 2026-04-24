# CLAUDE.md - Data Drift Detection System

## Project Overview

This is a **Data Drift Detection System** (資料漂移偵測系統) - a Python package for detecting and analyzing statistical drift in datasets. It's primarily used for monitoring machine learning model inputs in production environments.

**Purpose**: Detect when the statistical properties of production data deviate significantly from training/reference data, which can indicate model degradation or data quality issues.

**Language**: Mixed Chinese/English (Chinese comments and documentation, English code and variable names)

---

## Repository Structure

```
data_drift_module/                    ← git repo root
├── pyproject.toml                    # 套件定義，pip install -e . 可安裝
├── .gitignore
├── CLAUDE.md                         # 此文件
│
├── data_drift_module/                ← 可安裝的 Python 套件（flat layout）
│   ├── __init__.py                   # 公共 API
│   ├── detector.py                   # DataDriftDetector 偵測器
│   ├── visualizer.py                 # DriftVisualizer 視覺化
│   └── config.py                     # DriftDetectionConfig, ConfigSelector
│
└── examples/                         # 使用範例
    ├── quick_start.py                # 完整快速入門腳本
    └── notebooks/
        ├── quick_start_drift.ipynb   # Jupyter notebook - 漂移測試
        └── quick_start_nondrift.ipynb # Jupyter notebook - 無漂移測試
```

---

## Installation

```bash
pip install -e .
```

安裝後即可標準 import：

```python
from data_drift_module import DataDriftDetector, DriftVisualizer
from data_drift_module.config import DriftDetectionConfig, ConfigSelector
```

---

## Core Components

### 1. DataDriftDetector (`data_drift_module/detector.py`)

**Purpose**: 統一的漂移偵測器，合併了原有的基礎版與進階版

**Key Methods**:
- `__init__(reference_data, threshold=0.05, psi_threshold=0.2, jsd_threshold=0.1, min_sample_size=30, enable_mann_whitney=False, enable_anderson_darling=False)` - 初始化
- `detect_drift(current_data)` - 執行漂移偵測
- `generate_report(output_format='text'|'markdown')` - 生成報告
- `get_drifted_features()` - 取得漂移特徵列表

**Supported Tests**:

| 特徵類型 | 方法 | 角色 | 說明 |
|---------|------|------|------|
| 數值型 | KS Test | 主力 | 分布比較，非參數 |
| 數值型 | PSI | 主力 | 業界金融標準，可量化漂移程度 |
| 數值型 | JSD | 主力 | 有界 [0,1]，與 KS 互補 |
| 數值型 | Wasserstein Distance | 輔助 | 只記錄，不觸發漂移判定 |
| 數值型 | Mann-Whitney U | 可選 | `enable_mann_whitney=True` 啟用 |
| 數值型 | Anderson-Darling | 可選 | `enable_anderson_darling=True` 啟用 |
| 類別型 | Chi-Square Test | 主力 | 類別分布比較 |
| 類別型 | PSI（類別版） | 主力 | 類別比例偏移量化 |
| 類別型 | 新/消失類別偵測 | 主力 | 語義最清晰且零計算成本 |
| 附加 | 缺失值漂移 | 附加 | Z-test for proportions |
| 附加 | 樣本大小驗證 | 附加 | 不足時發出警告 |

### 2. DriftVisualizer (`data_drift_module/visualizer.py`)

**Purpose**: 漂移視覺化工具

**Constructor**: `DriftVisualizer(detector, current_data)` — 必須同時傳入兩個參數

**Key Methods**:
- `plot_drift_summary()` — 圓餅圖 + 特徵狀態條形圖 + 嚴重程度
- `plot_numerical_drift(feature_name)` — 數值型特徵完整分析（直方圖、箱型圖、Q-Q、CDF、統計量）
- `plot_categorical_drift(feature_name)` — 類別型特徵完整分析（條形圖、差異圖、圓餅圖）
- `plot_metrics_comparison()` — 多指標比較儀表板（PSI、漂移原因、類型分析、缺失值）
- `plot_all_drifts(save_dir)` — 生成所有圖表

### 3. Configuration System (`data_drift_module/config.py`)

**Key Classes**:

#### DriftDetectionConfig
靜態配置類別，包含預設閾值常數

**Configuration Methods**:
- `get_default_config()` — 標準設定
- `get_strict_config()` — 更敏感（較易偵測到漂移）
- `get_loose_config()` — 較寬鬆
- `get_production_config()` — 生產環境推薦設定

#### ConfigSelector
```python
config = ConfigSelector.select_by_scenario('production')
# 可選 scenario: 'development', 'production', 'monitoring', 'research', 'strict', 'loose'

config = ConfigSelector.select_by_data_size(n_samples)
```

---

## Statistical Methods Explained

### Numerical Feature Tests

1. **Kolmogorov-Smirnov (KS) Test**
   - 比較 CDF 的最大距離，返回 p-value（< threshold = 漂移）
   - 非參數，無分布假設

2. **PSI (Population Stability Index)**
   - 業界金融標準：< 0.1（穩定）、0.1-0.2（輕微）、> 0.2（顯著漂移）
   - 使用分位數分箱 + epsilon 平滑確保穩定性

3. **Jensen-Shannon Divergence (JSD)**
   - 有界 [0, 1]，對稱，基於 KL Divergence
   - 預設閾值 0.1，可透過 `jsd_threshold` 調整

4. **Wasserstein Distance** *(輔助)*
   - 只記錄於 tests 結果中，不影響 `has_drift` 判定
   - 相對於資料範圍解釋

5. **Mann-Whitney U Test** *(可選，預設關閉)*
   - 非參數中位數差異檢定，`enable_mann_whitney=True` 啟用

6. **Anderson-Darling Test** *(可選，預設關閉)*
   - `enable_anderson_darling=True` 啟用

### Categorical Feature Tests

1. **Chi-Square Test** — 類別分布獨立性檢定，返回 p-value
2. **PSI for Categorical** — 與數值型相同公式，套用類別比例
3. **新/消失類別偵測** — 直接比較兩資料集的類別集合

### Missing Value Detection

- Z-test for proportions 比較缺失率差異
- 同時滿足「p < threshold」且「缺失率差異 > 5%」才判定為漂移

---

## Development Workflows

### Typical Usage Pattern

```python
import pandas as pd
from data_drift_module import DataDriftDetector, DriftVisualizer
from data_drift_module.config import ConfigSelector

# 1. 載入資料
reference_data = pd.read_csv('data/training_data.csv')
current_data = pd.read_csv('data/production_data.csv')

# 2. 選擇配置
config = ConfigSelector.select_by_scenario('production')

# 3. 執行偵測
detector = DataDriftDetector(
    reference_data=reference_data,
    threshold=config['significance_level'],
    psi_threshold=config['psi_threshold'],
    jsd_threshold=config['jsd_threshold'],
    min_sample_size=config['min_sample_size']
)
results = detector.detect_drift(current_data)

# 4. 查看結果
print(results['summary'])
drifted = detector.get_drifted_features()

# 5. 生成報告
print(detector.generate_report('text'))
print(detector.generate_report('markdown'))

# 6. 視覺化
viz = DriftVisualizer(detector, current_data)
viz.plot_drift_summary()
viz.plot_metrics_comparison()
for feature in drifted:
    viz.plot_numerical_drift(feature)   # 或 plot_categorical_drift
```

---

## Key Conventions

### Code Style

1. **Language**:
   - 函式/類別名稱：English
   - 變數名稱：English
   - 註解/docstring：Traditional Chinese
   - 使用者訊息：Chinese

2. **Naming**:
   - Classes: PascalCase (`DataDriftDetector`)
   - Methods/functions: snake_case (`detect_drift`)
   - Private methods: leading underscore (`_detect_feature_drift`)
   - Constants: UPPER_SNAKE_CASE (`PSI_THRESHOLD_DEFAULT`)

3. **Docstrings**: Chinese，格式如下：
   ```python
   """
   簡短描述

   Parameters:
   -----------
   param_name : type
       參數說明

   Returns:
   --------
   type : 說明
   """
   ```

### Data Structure Conventions

#### Drift Detection Results Structure

```python
{
    'summary': {
        'total_features': int,
        'drifted_features': int,
        'drift_percentage': float,
        'warnings': list
    },
    'feature_results': {
        'feature_name': {
            'feature_name': str,
            'feature_type': 'numerical' | 'categorical',
            'has_drift': bool,
            'drift_reasons': list,   # e.g. ['ks_test', 'psi', 'jsd']
            'tests': {
                # 數值型: ks_test, psi, jensen_shannon_divergence, wasserstein_distance
                #   可選: mann_whitney, anderson_darling
                # 類別型: chi_square_test, psi
            },
            'statistics': {...},
            'missing_analysis': {...},
            'sample_check': {...}
        }
    }
}
```

---

## Decision Thresholds and Interpretations

### PSI Interpretation

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.1 | 無顯著變化 | 繼續監控 |
| 0.1 - 0.2 | 輕微變化 | 調查 |
| > 0.2 | 顯著漂移 | 考慮重新訓練 |

### Drift Percentage Interpretation

| Drift % | Severity | Actions |
|---------|----------|---------|
| < 5% | 穩定 | 正常監控 |
| 5-15% | 輕微漂移 | 持續監控，記錄趨勢 |
| 15-30% | 中度漂移 | 密切監控，分析主要特徵，準備重新訓練 |
| ≥ 30% | 嚴重漂移 | 立即行動：檢查資料管道、評估模型、重新訓練 |

---

## Extension Points

### Adding New Statistical Tests

```python
def _calculate_new_metric(self, ref_data, curr_data):
    return metric_value

def _detect_numerical_drift(self, ...):
    # 在現有測試後加入
    if self.enable_new_metric:
        new_value = self._calculate_new_metric(ref_data, curr_data)
        if new_value > threshold:
            drift_reasons.append('new_metric')
        tests['new_metric'] = {'value': float(new_value), 'interpretation': ...}
```

### Custom Configuration Profiles

```python
@classmethod
def get_custom_config(cls) -> dict:
    return {
        'significance_level': 0.03,
        'psi_threshold': 0.15,
        'jsd_threshold': 0.08,
        ...
    }
```

---

## AI Assistant Guidelines

### When Modifying Code

1. **Preserve Language Pattern** — Chinese comments/docstrings, English code
2. **Maintain Results Structure** — `drift_results` dict 格式不要破壞
3. **Test Both Feature Types** — 數值型和類別型都要驗證
4. **Update CLAUDE.md** — 重要變更後更新此文件

### When Analyzing Drift Results

1. 使用多指標綜合判斷（不要依賴單一指標）
2. PSI > 0.2 且多個檢定失敗 = 高信心漂移
3. 小樣本警告時結果可靠性較低（檢查 `sample_check`）
4. 注意 `missing_analysis` 是否有資料品質問題

---

## Quick Reference

### Import Statements

```python
from data_drift_module import DataDriftDetector, DriftVisualizer
from data_drift_module.config import DriftDetectionConfig, ConfigSelector
```

### Minimal Working Example

```python
import pandas as pd
from data_drift_module import DataDriftDetector

ref = pd.read_csv('reference.csv')
curr = pd.read_csv('current.csv')

detector = DataDriftDetector(ref)
results = detector.detect_drift(curr)

print(f"Drift detected in {results['summary']['drifted_features']} features")
print(detector.generate_report())
```

### File to Use For Each Task

| Task | File |
|------|------|
| 漂移偵測 | `data_drift_module/detector.py` |
| 視覺化 | `data_drift_module/visualizer.py` |
| 配置管理 | `data_drift_module/config.py` |
| 快速入門 | `examples/quick_start.py` |
| Jupyter 範例 | `examples/notebooks/` |

---

**Last Updated**: 2026-04-24
**Version**: 1.0.0
