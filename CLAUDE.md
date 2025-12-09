# CLAUDE.md - Data Drift Detection System

## Project Overview

This is a **Data Drift Detection System** (資料漂移偵測系統) - a comprehensive Python-based toolkit for detecting and analyzing statistical drift in datasets. It's primarily used for monitoring machine learning model inputs in production environments.

**Purpose**: Detect when the statistical properties of production data deviate significantly from training/reference data, which can indicate model degradation or data quality issues.

**Language**: Mixed Chinese/English (Chinese comments and documentation, English code and variable names)

---

## Repository Structure

```
data_drift_module/
├── dependencies/              # Core modules (basic implementation)
│   ├── __init__.py           # Package initialization - exports DataDriftDetector and DriftVisualizer
│   ├── data_drift_detector.py    # Basic drift detector with KS, Chi-Square, PSI, Wasserstein
│   └── drift_visualizer.py       # Basic visualization tools
│
├── files/                    # Improved/enhanced implementations
│   ├── improved_data_drift_detector.py  # Enhanced detector with additional tests
│   ├── improved_drift_visualizer.py     # Enhanced visualizer with more charts
│   ├── drift_config.py                  # Configuration management system
│   ├── quick_start_improved.py          # Comprehensive quick start script
│   └── quick_start_simple.py            # Simple quick start example
│
├── script/                   # Usage examples and scripts
│   ├── quick_start.py        # Main quick start script
│   ├── quick_start_drift.ipynb          # Jupyter notebook - drift testing
│   └── quick_start_nonfrift.ipynb       # Jupyter notebook - no-drift testing
│
├── .gitognore                # Git ignore file (NOTE: filename has typo)
└── CLAUDE.md                 # This file
```

---

## Core Components

### 1. DataDriftDetector (Basic - `dependencies/data_drift_detector.py`)

**Purpose**: Detect drift using fundamental statistical tests

**Key Methods**:
- `__init__(reference_data, threshold=0.05)` - Initialize with baseline data
- `detect_drift(current_data)` - Run drift detection on new data
- `generate_report(output_format='text'|'markdown')` - Generate reports
- `get_drifted_features()` - Get list of features with detected drift

**Supported Tests**:
- **Numerical Features**:
  - Kolmogorov-Smirnov (KS) Test - distribution comparison
  - Wasserstein Distance - distance between distributions
  - Population Stability Index (PSI) - shift measurement
- **Categorical Features**:
  - Chi-Square Test - independence test
  - PSI for categorical - category distribution shift

**Configuration**:
- `threshold` (float): Significance level for statistical tests (default: 0.05)

### 2. ImprovedDataDriftDetector (Enhanced - `files/improved_data_drift_detector.py`)

**Purpose**: Advanced drift detection with additional metrics and validations

**Additional Features**:
- Missing value drift detection
- Sample size validation
- More statistical tests
- Improved PSI calculation with better stability

**Additional Tests**:
- **Numerical**:
  - Jensen-Shannon Divergence (JSD)
  - Mann-Whitney U Test
  - Anderson-Darling Test (reference only)
- **Categorical**:
  - Total Variation Distance (TVD)
  - Hellinger Distance
  - New/disappeared category detection

**Configuration Parameters**:
- `threshold` (float): Significance level (default: 0.05)
- `min_sample_size` (int): Minimum samples required (default: 30)
- `psi_threshold` (float): PSI drift threshold (default: 0.2)

**New in Results**:
- `missing_analysis`: Missing value drift information
- `sample_check`: Sample size validation
- `drift_reasons`: List of which tests detected drift

### 3. DriftVisualizer (Basic - `dependencies/drift_visualizer.py`)

**Purpose**: Create visualizations for drift analysis

**Key Methods**:
- `plot_drift_summary()` - Overview charts
- `plot_numerical_drift(feature_name)` - Numerical feature analysis
- `plot_categorical_drift(feature_name)` - Categorical feature analysis
- `plot_psi_heatmap()` - PSI values across features
- `plot_all_drifts(save_dir)` - Generate all visualizations

**Chart Types**:
- Pie charts (drift ratios)
- Bar charts (feature-wise drift status)
- Histograms (distribution comparison)
- Box plots (statistical comparison)

### 4. ImprovedDriftVisualizer (Enhanced - `files/improved_drift_visualizer.py`)

**Purpose**: Advanced visualizations with more comprehensive charts

**Important**: Requires `current_data` parameter in constructor (fixed from basic version)

```python
visualizer = ImprovedDriftVisualizer(detector, current_data)
```

**Additional Visualizations**:
- Q-Q plots for distribution comparison
- CDF (Cumulative Distribution Function) plots
- Drift severity indicators
- Missing value change charts
- Multi-metric comparison dashboard
- Drift reason distribution

**New Methods**:
- `plot_metrics_comparison()` - Compare all metrics across features
- Enhanced numerical/categorical drift plots with more subplots

### 5. Configuration System (`files/drift_config.py`)

**Purpose**: Centralized configuration management

**Key Classes**:

#### DriftDetectionConfig
Static configuration class with predefined thresholds

**Configuration Methods**:
- `get_default_config()` - Standard settings
- `get_strict_config()` - More sensitive to drift
- `get_loose_config()` - Less sensitive to drift
- `get_production_config()` - Recommended for production

**Key Thresholds**:
```python
SIGNIFICANCE_LEVEL = 0.05           # p-value threshold
PSI_THRESHOLD_DEFAULT = 0.2         # PSI drift threshold
JSD_THRESHOLD_MODERATE = 0.1        # JSD threshold
TVD_THRESHOLD_MODERATE = 0.2        # TVD threshold
HELLINGER_THRESHOLD_MODERATE = 0.2  # Hellinger threshold
MIN_SAMPLE_SIZE = 30                # Minimum samples
```

#### ConfigSelector
Utility for selecting configurations based on scenario or data size

**Methods**:
- `select_by_scenario(scenario)` - Choose config by use case
  - Available scenarios: 'development', 'production', 'monitoring', 'research', 'strict', 'loose'
- `select_by_data_size(n_samples)` - Adjust config based on sample size

---

## Statistical Methods Explained

### Numerical Feature Tests

1. **Kolmogorov-Smirnov (KS) Test**
   - Measures maximum distance between CDFs
   - Returns p-value (< threshold = drift detected)
   - Non-parametric, no distribution assumptions

2. **Wasserstein Distance**
   - Measures "earth mover's distance" between distributions
   - Higher values indicate more difference
   - Interpretable in original data units

3. **Population Stability Index (PSI)**
   - Industry-standard metric for distribution shift
   - Thresholds: < 0.1 (stable), 0.1-0.2 (slight change), > 0.2 (significant drift)
   - Formula: PSI = Σ(actual% - expected%) × ln(actual% / expected%)

4. **Jensen-Shannon Divergence (JSD)**
   - Symmetric measure of distribution similarity
   - Range: [0, 1], where 0 = identical distributions
   - Based on KL divergence but symmetric and bounded

5. **Mann-Whitney U Test**
   - Non-parametric test for distribution differences
   - Tests if distributions have same median
   - Returns p-value like KS test

### Categorical Feature Tests

1. **Chi-Square Test**
   - Tests independence of categorical distributions
   - Returns p-value (< threshold = drift detected)
   - Requires sufficient samples in each category

2. **Total Variation Distance (TVD)**
   - Measures maximum probability difference
   - Formula: TVD = 0.5 × Σ|P(x) - Q(x)|
   - Range: [0, 1]

3. **Hellinger Distance**
   - Based on Bhattacharyya coefficient
   - Symmetric measure of overlap
   - Range: [0, 1], 0 = identical distributions

4. **PSI for Categorical**
   - Same formula as numerical but applied to category proportions
   - Detects changes in category distributions

### Missing Value Detection

- Z-test for proportions to detect significant changes in missing rates
- Pooled proportion method for standard error calculation
- Reports both absolute change and statistical significance

---

## Development Workflows

### Typical Usage Pattern

1. **Load Data**
   ```python
   import pandas as pd
   reference_data = pd.read_csv('training_data.csv')
   current_data = pd.read_csv('production_data.csv')
   ```

2. **Preprocess** (optional)
   ```python
   # Remove irrelevant columns
   columns_to_drop = ['id', 'timestamp', 'user_id']
   reference_data = reference_data.drop(columns=columns_to_drop)
   current_data = current_data.drop(columns=columns_to_drop)
   ```

3. **Choose Implementation**
   - Use `dependencies/` modules for basic, fast detection
   - Use `files/improved_*` modules for comprehensive analysis

4. **Configure Detection**
   ```python
   from drift_config import ConfigSelector
   config = ConfigSelector.select_by_scenario('production')
   ```

5. **Run Detection**
   ```python
   from improved_data_drift_detector import ImprovedDataDriftDetector

   detector = ImprovedDataDriftDetector(
       reference_data=reference_data,
       threshold=config['significance_level'],
       min_sample_size=config['min_sample_size'],
       psi_threshold=config['psi_threshold']
   )

   results = detector.detect_drift(current_data)
   ```

6. **Analyze Results**
   ```python
   # Check summary
   print(results['summary'])

   # Get drifted features
   drifted = detector.get_drifted_features()

   # Generate reports
   text_report = detector.generate_report('text')
   md_report = detector.generate_report('markdown')
   ```

7. **Visualize** (optional)
   ```python
   from improved_drift_visualizer import ImprovedDriftVisualizer

   viz = ImprovedDriftVisualizer(detector, current_data)
   viz.plot_drift_summary()
   viz.plot_metrics_comparison()

   # Individual feature analysis
   for feature in drifted:
       viz.plot_numerical_drift(feature)
   ```

### Quick Start Scripts

Two quick start options provided:

1. **Simple** (`files/quick_start_simple.py`)
   - Uses basic detector
   - Minimal configuration
   - ~200 lines, easy to understand
   - Good for learning/prototyping

2. **Improved** (`files/quick_start_improved.py`)
   - Uses enhanced detector and visualizer
   - Comprehensive analysis pipeline
   - ~515 lines with detailed reporting
   - Production-ready with decision logic
   - Includes 12 steps: data loading → visualization → decision recommendations

---

## Key Conventions

### Code Style

1. **Language**:
   - All function/class names in English
   - All variable names in English
   - Comments and docstrings in Chinese (Traditional)
   - User-facing messages in Chinese

2. **Naming**:
   - Classes: PascalCase (e.g., `DataDriftDetector`)
   - Methods/functions: snake_case (e.g., `detect_drift`)
   - Private methods: leading underscore (e.g., `_detect_feature_drift`)
   - Constants: UPPER_SNAKE_CASE (e.g., `PSI_THRESHOLD_DEFAULT`)

3. **Docstrings**:
   - Chinese-language docstrings with English parameter names
   - Format:
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
        'warnings': list  # Only in ImprovedDataDriftDetector
    },
    'feature_results': {
        'feature_name': {
            'feature_name': str,
            'feature_type': 'numerical' | 'categorical',
            'has_drift': bool,
            'drift_reasons': list,  # Only in improved version
            'tests': {
                # Test-specific results
                'ks_test': {'statistic': float, 'p_value': float, 'interpretation': str},
                'psi': {'value': float, 'interpretation': str},
                # ... more tests
            },
            'statistics': {
                'reference': {...},
                'current': {...}
            },
            'missing_analysis': {...},  # Only in improved version
            'sample_check': {...}       # Only in improved version
        }
    }
}
```

### File Organization

- **Core logic**: `dependencies/` - stable, well-tested implementations
- **Experimental/Enhanced**: `files/` - newer features, improvements
- **Examples**: `script/` - usage demonstrations
- **Outputs**: Not committed (in .gitignore as `output/`, `data/`)

### Testing Approach

Testing done via Jupyter notebooks in `script/`:
- `quick_start_drift.ipynb` - Tests with known drift data
- `quick_start_nonfrift.ipynb` - Tests with stable data

---

## Environment Setup

### Dependencies

Required Python packages (inferred from imports):
```
pandas
numpy
scipy
matplotlib
seaborn
python-dotenv
```

### Configuration via .env

Scripts expect `.env` file with:
```
proj_path=/path/to/data_drift_module
```

Used in quick start scripts:
```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os
proj_path = os.getenv('proj_path')
```

### Data Expectations

Scripts expect data in `data/` directory (gitignored):
- `training_data.csv` - Reference/baseline data
- `production_data_with_drift.csv` - Test data with drift
- `production_data_no_drift.csv` - Test data without drift

---

## Known Issues and Gotchas

### 1. Gitignore Filename Typo

**Issue**: The gitignore file is named `.gitognore` (typo: missing 'i')
**Impact**: File is not being used by Git
**Location**: `/home/user/data_drift_module/.gitognore`
**Fix**: Should be renamed to `.gitignore`

### 2. Current Data in Visualizer

**Issue**: Basic `DriftVisualizer` doesn't store `current_data`, limiting some visualizations
**Status**: Fixed in `ImprovedDriftVisualizer` which requires it as constructor parameter
**Usage**: Always use `ImprovedDriftVisualizer(detector, current_data)` with both parameters

### 3. Chi-Square Test Sensitivity

**Issue**: Chi-square test can fail with small sample sizes or rare categories
**Mitigation**:
- Code includes try-except blocks to handle failures
- Sample size warnings in improved version
- Frequency normalization to ensure valid counts

### 4. PSI Calculation Stability

**Issue**: PSI can produce NaN/Inf with zero-frequency bins
**Mitigation**:
- Improved version uses epsilon smoothing (default: 1e-4)
- Unique breakpoint handling
- Fallback to 0.0 on calculation errors

### 5. Missing Data Handling

**Issue**: Missing values handled by dropping (`.dropna()`)
**Impact**: Separate missing value drift detection added in improved version
**Recommendation**: Check `missing_analysis` field in results

---

## Decision Thresholds and Interpretations

### PSI Interpretation

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.1 | No significant change | Continue monitoring |
| 0.1 - 0.2 | Slight change | Investigate |
| > 0.2 | Significant drift | Consider retraining |

### Drift Percentage Interpretation

Based on `quick_start_improved.py` decision logic:

| Drift % | Severity | Actions |
|---------|----------|---------|
| < 5% | Stable | Normal monitoring |
| 5-15% | Light drift | Continuous monitoring, trend logging |
| 15-30% | Moderate drift | Close monitoring, analyze main features, prepare retraining |
| ≥ 30% | Severe drift | Immediate action: check data pipeline, analyze causes, evaluate model, retrain |

### Statistical Test Thresholds

**Standard Configuration**:
- Significance level: 0.05 (p-value threshold)
- PSI threshold: 0.2
- JSD threshold: 0.1
- TVD threshold: 0.2
- Hellinger threshold: 0.2
- Missing rate difference: 0.05 (5%)

**Strict Configuration** (for production monitoring):
- Significance level: 0.01
- PSI threshold: 0.1
- Lower thresholds for all distance metrics

---

## Extension Points

### Adding New Statistical Tests

1. Add test function to detector class:
   ```python
   def _calculate_new_metric(self, ref_data, curr_data):
       # Implementation
       return metric_value
   ```

2. Add to appropriate drift detection method:
   ```python
   def _detect_numerical_drift(self, ...):
       # ... existing tests
       new_metric = self._calculate_new_metric(ref_data, curr_data)
       if new_metric > threshold:
           drift_reasons.append('new_metric')
   ```

3. Add interpretation function:
   ```python
   def _interpret_new_metric(self, value):
       if value < 0.1:
           return "低風險"
       # ... more conditions
   ```

4. Include in test results dictionary

### Adding New Visualizations

1. Add method to visualizer class:
   ```python
   def plot_new_visualization(self, feature_name, figsize, save_path):
       # Create matplotlib figure
       # Add to ImprovedDriftVisualizer
   ```

2. Include in `plot_all_drifts()` if applicable

### Custom Configuration Profiles

Add to `DriftDetectionConfig` class:
```python
@classmethod
def get_custom_config(cls) -> dict:
    return {
        'significance_level': 0.03,
        'psi_threshold': 0.15,
        # ... other parameters
    }
```

---

## AI Assistant Guidelines

### When Modifying Code

1. **Preserve Language Pattern**:
   - Keep Chinese comments/docstrings
   - Keep English code/variable names
   - Maintain existing docstring format

2. **Maintain Compatibility**:
   - Both basic and improved versions should work independently
   - Don't break existing quick start scripts
   - Preserve results dictionary structure

3. **Testing**:
   - Test with both numerical and categorical features
   - Verify with small sample sizes
   - Check edge cases (all same values, many missing values)

4. **Documentation**:
   - Update this CLAUDE.md for significant changes
   - Update method docstrings
   - Update quick start scripts if API changes

### When Analyzing Drift Results

1. **Check Multiple Indicators**:
   - Don't rely on single test
   - PSI is industry standard but check p-values too
   - Improved version provides drift_reasons list

2. **Consider Context**:
   - Sample size matters (check sample_check field)
   - Missing value changes can indicate data quality issues
   - New/disappeared categories in categorical features

3. **Interpretation Priority**:
   - PSI > 0.2 with multiple test failures = high confidence drift
   - Single test failure = investigate further
   - Small sample warnings = results less reliable

### Common Tasks

#### Adding a New Feature to Track

```python
# No code change needed - just ensure feature in both datasets
reference_data['new_feature'] = ...
current_data['new_feature'] = ...
# Detector will automatically detect and analyze
```

#### Adjusting Sensitivity

```python
# Option 1: Direct configuration
detector = ImprovedDataDriftDetector(
    reference_data=ref_data,
    threshold=0.01,  # More strict
    psi_threshold=0.15  # More strict
)

# Option 2: Use config profiles
config = DriftDetectionConfig.get_strict_config()
detector = ImprovedDataDriftDetector(
    reference_data=ref_data,
    threshold=config['significance_level'],
    psi_threshold=config['psi_threshold']
)
```

#### Generating Custom Reports

```python
# Access raw results
results = detector.detect_drift(current_data)

# Build custom analysis
for feature, result in results['feature_results'].items():
    if result['has_drift']:
        # Custom logic here
        tests = result['tests']
        stats = result['statistics']
        # ... analyze as needed
```

---

## Project History and Evolution

Based on commit history:
1. Initial commit - basic structure
2. Added .gitignore (with typo)
3. Multiple fixes to .gitignore patterns
4. Removed ignored files from repository

**Recent Focus**: Repository cleanup and gitignore configuration

---

## Future Enhancement Ideas

Potential improvements not yet implemented:

1. **Automated Monitoring**:
   - Scheduled drift detection
   - Alert system integration
   - Dashboard for continuous monitoring

2. **Additional Tests**:
   - Cramér's V for categorical associations
   - Effect size metrics
   - Multivariate drift detection

3. **Performance Optimization**:
   - Parallel processing for multiple features
   - Sampling for very large datasets (partially implemented in config)
   - Incremental/online drift detection

4. **Better Reporting**:
   - HTML report generation
   - Interactive visualizations (Plotly)
   - Exportable drift dashboards

5. **Integration**:
   - MLflow integration
   - Model registry integration
   - CI/CD pipeline integration

---

## Quick Reference

### Import Statements

```python
# Basic version
from dependencies import DataDriftDetector, DriftVisualizer

# Improved version
from files.improved_data_drift_detector import ImprovedDataDriftDetector
from files.improved_drift_visualizer import ImprovedDriftVisualizer
from files.drift_config import DriftDetectionConfig, ConfigSelector
```

### Minimal Working Example

```python
import pandas as pd
from improved_data_drift_detector import ImprovedDataDriftDetector

# Load data
ref = pd.read_csv('reference.csv')
curr = pd.read_csv('current.csv')

# Detect
detector = ImprovedDataDriftDetector(ref)
results = detector.detect_drift(curr)

# Report
print(f"Drift detected in {results['summary']['drifted_features']} features")
print(detector.generate_report())
```

### File to Use For Each Task

| Task | File to Use |
|------|-------------|
| Basic drift detection | `dependencies/data_drift_detector.py` |
| Comprehensive detection | `files/improved_data_drift_detector.py` |
| Basic visualization | `dependencies/drift_visualizer.py` |
| Advanced visualization | `files/improved_drift_visualizer.py` |
| Configuration | `files/drift_config.py` |
| Quick start / learning | `files/quick_start_simple.py` |
| Production pipeline | `files/quick_start_improved.py` |

---

## Contact and Contribution

This is a self-contained project for data drift detection. When contributing:

1. Follow existing code style and language patterns
2. Test with both basic and improved versions
3. Update this CLAUDE.md for significant changes
4. Add examples to quick start scripts when adding features
5. Fix the .gitignore filename typo if touching that area

---

**Last Updated**: 2025-12-09
**Version**: Based on commit 02a6eb3
