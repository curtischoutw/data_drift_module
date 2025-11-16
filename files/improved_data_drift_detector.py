"""
資料漂移偵測系統 (Data Drift Detection System) - 改進版
支援多種統計檢定方法、進階指標和視覺化功能
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ImprovedDataDriftDetector:
    """
    改進版資料漂移偵測器
    - 支援更多統計檢定方法
    - 加入樣本大小檢查
    - 加入缺失值漂移檢測
    - 改進的穩定性
    """
    
    def __init__(self, 
                 reference_data: pd.DataFrame, 
                 threshold: float = 0.05,
                 min_sample_size: int = 30,
                 psi_threshold: float = 0.2):
        """
        初始化偵測器
        
        Parameters:
        -----------
        reference_data : pd.DataFrame
            基準資料集
        threshold : float
            顯著性水準 (預設 0.05)
        min_sample_size : int
            最小樣本數要求 (預設 30)
        psi_threshold : float
            PSI 顯著漂移閾值 (預設 0.2)
        """
        self.reference_data = reference_data
        self.threshold = threshold
        self.min_sample_size = min_sample_size
        self.psi_threshold = psi_threshold
        self.drift_results = {}
        
    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        偵測資料漂移
        
        Parameters:
        -----------
        current_data : pd.DataFrame
            當前資料集
            
        Returns:
        --------
        Dict : 包含所有特徵的漂移偵測結果
        """
        results = {
            'summary': {
                'total_features': len(self.reference_data.columns),
                'drifted_features': 0,
                'drift_percentage': 0.0,
                'warnings': []
            },
            'feature_results': {}
        }
        
        for column in self.reference_data.columns:
            if column not in current_data.columns:
                warning_msg = f"警告: 欄位 '{column}' 不存在於當前資料中"
                print(warning_msg)
                results['summary']['warnings'].append(warning_msg)
                continue
                
            feature_result = self._detect_feature_drift(
                column, 
                self.reference_data[column], 
                current_data[column]
            )
            
            results['feature_results'][column] = feature_result
            
            if feature_result['has_drift']:
                results['summary']['drifted_features'] += 1
        
        results['summary']['drift_percentage'] = (
            results['summary']['drifted_features'] / 
            results['summary']['total_features'] * 100
        )
        
        self.drift_results = results
        return results
    
    def _detect_feature_drift(self, column_name: str, 
                             ref_data: pd.Series, 
                             curr_data: pd.Series) -> Dict:
        """
        偵測單一特徵的漂移
        """
        # 檢測缺失值漂移
        missing_result = self._detect_missing_drift(ref_data, curr_data)
        
        # 移除缺失值
        ref_clean = ref_data.dropna()
        curr_clean = curr_data.dropna()
        
        # 樣本大小檢查
        sample_check = self._check_sample_size(ref_clean, curr_clean)
        
        # 判斷資料類型並進行漂移檢測
        if self._is_numerical(ref_clean):
            result = self._detect_numerical_drift(column_name, ref_clean, curr_clean)
        else:
            result = self._detect_categorical_drift(column_name, ref_clean, curr_clean)
        
        # 加入缺失值和樣本檢查結果
        result['missing_analysis'] = missing_result
        result['sample_check'] = sample_check
        
        # 綜合判斷是否有漂移 (考慮缺失值漂移)
        if missing_result['has_drift']:
            result['has_drift'] = True
            result['drift_reasons'] = result.get('drift_reasons', [])
            result['drift_reasons'].append('missing_value_drift')
        
        return result
    
    def _is_numerical(self, data: pd.Series) -> bool:
        """判斷是否為數值型資料"""
        return pd.api.types.is_numeric_dtype(data)
    
    def _check_sample_size(self, ref_data: pd.Series, 
                          curr_data: pd.Series) -> Dict:
        """
        檢查樣本大小是否足夠進行統計檢定
        """
        ref_size = len(ref_data)
        curr_size = len(curr_data)
        
        warnings = []
        if ref_size < self.min_sample_size:
            warnings.append(f"參考資料樣本數過小: {ref_size} < {self.min_sample_size}")
        if curr_size < self.min_sample_size:
            warnings.append(f"當前資料樣本數過小: {curr_size} < {self.min_sample_size}")
        
        return {
            'is_valid': len(warnings) == 0,
            'ref_size': ref_size,
            'curr_size': curr_size,
            'warnings': warnings
        }
    
    def _detect_missing_drift(self, ref_data: pd.Series, 
                             curr_data: pd.Series) -> Dict:
        """
        檢測缺失值的漂移
        """
        ref_missing_rate = ref_data.isna().mean()
        curr_missing_rate = curr_data.isna().mean()
        
        missing_diff = abs(curr_missing_rate - ref_missing_rate)
        
        # 簡單的比例差異檢定
        # 使用 Z-test for proportions
        try:
            ref_missing_count = ref_data.isna().sum()
            curr_missing_count = curr_data.isna().sum()
            ref_total = len(ref_data)
            curr_total = len(curr_data)
            
            # 合併比例
            pooled_prop = (ref_missing_count + curr_missing_count) / (ref_total + curr_total)
            
            # 標準誤
            se = np.sqrt(pooled_prop * (1 - pooled_prop) * (1/ref_total + 1/curr_total))
            
            # Z 統計量
            if se > 0:
                z_stat = (curr_missing_rate - ref_missing_rate) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                z_stat = 0.0
                p_value = 1.0
                
        except:
            z_stat = 0.0
            p_value = 1.0
        
        return {
            'reference_missing_rate': float(ref_missing_rate),
            'current_missing_rate': float(curr_missing_rate),
            'difference': float(missing_diff),
            'z_statistic': float(z_stat),
            'p_value': float(p_value),
            'has_drift': p_value < self.threshold and missing_diff > 0.05
        }
    
    def _detect_numerical_drift(self, column_name: str,
                               ref_data: pd.Series, 
                               curr_data: pd.Series) -> Dict:
        """
        偵測連續型變數的漂移
        使用多種統計檢定和距離度量
        """
        drift_reasons = []
        
        # 1. KS 檢定
        ks_statistic, ks_pvalue = stats.ks_2samp(ref_data, curr_data)
        if ks_pvalue < self.threshold:
            drift_reasons.append('ks_test')
        
        # 2. Wasserstein Distance
        wasserstein_dist = stats.wasserstein_distance(ref_data, curr_data)
        
        # 3. PSI (Population Stability Index)
        psi_value = self._calculate_psi_improved(ref_data, curr_data)
        if psi_value > self.psi_threshold:
            drift_reasons.append('psi')
        
        # 4. Jensen-Shannon Divergence (新增)
        jsd_value = self._calculate_jsd(ref_data, curr_data)
        if jsd_value > 0.1:  # JSD 閾值
            drift_reasons.append('jsd')
        
        # 5. Mann-Whitney U Test (新增)
        try:
            mw_statistic, mw_pvalue = stats.mannwhitneyu(
                ref_data, curr_data, alternative='two-sided'
            )
        except:
            mw_statistic, mw_pvalue = 0.0, 1.0
        
        if mw_pvalue < self.threshold:
            drift_reasons.append('mann_whitney')
        
        # 6. Anderson-Darling Test (新增,僅供參考)
        try:
            # 合併資料並標記來源
            combined = np.concatenate([ref_data, curr_data])
            labels = np.concatenate([
                np.zeros(len(ref_data)), 
                np.ones(len(curr_data))
            ])
            ad_result = stats.anderson_ksamp([ref_data, curr_data])
            ad_statistic = ad_result.statistic
            ad_pvalue = ad_result.significance_level if hasattr(ad_result, 'significance_level') else None
        except:
            ad_statistic = 0.0
            ad_pvalue = None
        
        # 統計摘要
        stats_summary = {
            'reference': {
                'mean': float(ref_data.mean()),
                'std': float(ref_data.std()),
                'median': float(ref_data.median()),
                'min': float(ref_data.min()),
                'max': float(ref_data.max()),
                'q25': float(ref_data.quantile(0.25)),
                'q75': float(ref_data.quantile(0.75)),
                'skewness': float(ref_data.skew()),
                'kurtosis': float(ref_data.kurtosis())
            },
            'current': {
                'mean': float(curr_data.mean()),
                'std': float(curr_data.std()),
                'median': float(curr_data.median()),
                'min': float(curr_data.min()),
                'max': float(curr_data.max()),
                'q25': float(curr_data.quantile(0.25)),
                'q75': float(curr_data.quantile(0.75)),
                'skewness': float(curr_data.skew()),
                'kurtosis': float(curr_data.kurtosis())
            }
        }
        
        return {
            'feature_name': column_name,
            'feature_type': 'numerical',
            'has_drift': len(drift_reasons) > 0,
            'drift_reasons': drift_reasons,
            'tests': {
                'ks_test': {
                    'statistic': float(ks_statistic),
                    'p_value': float(ks_pvalue),
                    'interpretation': 'Drift detected' if ks_pvalue < self.threshold else 'No drift'
                },
                'wasserstein_distance': {
                    'value': float(wasserstein_dist),
                    'interpretation': self._interpret_wasserstein(wasserstein_dist, ref_data, curr_data)
                },
                'psi': {
                    'value': float(psi_value),
                    'interpretation': self._interpret_psi(psi_value)
                },
                'jensen_shannon_divergence': {
                    'value': float(jsd_value),
                    'interpretation': self._interpret_jsd(jsd_value)
                },
                'mann_whitney': {
                    'statistic': float(mw_statistic),
                    'p_value': float(mw_pvalue),
                    'interpretation': 'Drift detected' if mw_pvalue < self.threshold else 'No drift'
                },
                'anderson_darling': {
                    'statistic': float(ad_statistic),
                    'interpretation': 'Available for reference'
                }
            },
            'statistics': stats_summary
        }
    
    def _detect_categorical_drift(self, column_name: str,
                                  ref_data: pd.Series, 
                                  curr_data: pd.Series) -> Dict:
        """
        偵測類別型變數的漂移
        使用多種統計檢定
        """
        drift_reasons = []
        
        # 取得所有類別
        all_categories = sorted(set(ref_data.unique()) | set(curr_data.unique()))
        
        # 計算頻率
        ref_counts = ref_data.value_counts()
        curr_counts = curr_data.value_counts()
        
        # 1. Chi-Square Test (修正版)
        ref_freq = np.array([ref_counts.get(cat, 0) for cat in all_categories])
        curr_freq = np.array([curr_counts.get(cat, 0) for cat in all_categories])
        
        # 使用列聯表方式
        contingency_table = np.array([ref_freq, curr_freq])
        try:
            chi2_stat, chi2_pvalue, dof, expected = stats.chi2_contingency(contingency_table)
        except:
            chi2_stat, chi2_pvalue = 0.0, 1.0
        
        if chi2_pvalue < self.threshold:
            drift_reasons.append('chi_square')
        
        # 2. PSI for categorical
        psi_value = self._calculate_categorical_psi(ref_data, curr_data)
        if psi_value > self.psi_threshold:
            drift_reasons.append('psi')
        
        # 3. Total Variation Distance (新增)
        tvd_value = self._calculate_tvd(ref_data, curr_data)
        if tvd_value > 0.2:  # TVD 閾值
            drift_reasons.append('tvd')
        
        # 4. Hellinger Distance (新增)
        hellinger_value = self._calculate_hellinger(ref_data, curr_data)
        if hellinger_value > 0.2:  # Hellinger 閾值
            drift_reasons.append('hellinger')
        
        # 5. 檢測新類別
        new_categories = set(curr_data.unique()) - set(ref_data.unique())
        disappeared_categories = set(ref_data.unique()) - set(curr_data.unique())
        
        if len(new_categories) > 0 or len(disappeared_categories) > 0:
            drift_reasons.append('category_change')
        
        return {
            'feature_name': column_name,
            'feature_type': 'categorical',
            'has_drift': len(drift_reasons) > 0,
            'drift_reasons': drift_reasons,
            'tests': {
                'chi_square_test': {
                    'statistic': float(chi2_stat),
                    'p_value': float(chi2_pvalue),
                    'interpretation': 'Drift detected' if chi2_pvalue < self.threshold else 'No drift'
                },
                'psi': {
                    'value': float(psi_value),
                    'interpretation': self._interpret_psi(psi_value)
                },
                'total_variation_distance': {
                    'value': float(tvd_value),
                    'interpretation': self._interpret_tvd(tvd_value)
                },
                'hellinger_distance': {
                    'value': float(hellinger_value),
                    'interpretation': self._interpret_hellinger(hellinger_value)
                }
            },
            'statistics': {
                'reference_categories': len(ref_data.unique()),
                'current_categories': len(curr_data.unique()),
                'new_categories': list(new_categories),
                'disappeared_categories': list(disappeared_categories),
                'reference_distribution': ref_data.value_counts().to_dict(),
                'current_distribution': curr_data.value_counts().to_dict()
            }
        }
    
    def _calculate_psi_improved(self, ref_data: pd.Series, curr_data: pd.Series, 
                               bins: int = 10, epsilon: float = 1e-4) -> float:
        """
        改進的 PSI 計算,提升穩定性
        """
        try:
            breakpoints = np.percentile(ref_data, np.linspace(0, 100, bins + 1))
            breakpoints = np.unique(breakpoints)
            
            if len(breakpoints) < 2:
                return 0.0
            
            ref_counts = np.histogram(ref_data, bins=breakpoints)[0]
            curr_counts = np.histogram(curr_data, bins=breakpoints)[0]
            
            # 改進的平滑處理
            ref_props = (ref_counts + epsilon) / (len(ref_data) + epsilon * len(breakpoints))
            curr_props = (curr_counts + epsilon) / (len(curr_data) + epsilon * len(breakpoints))
            
            # 計算 PSI
            psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
            
            return abs(psi)
        except:
            return 0.0
    
    def _calculate_categorical_psi(self, ref_data: pd.Series, 
                                   curr_data: pd.Series,
                                   epsilon: float = 1e-4) -> float:
        """
        計算類別型變數的 PSI
        """
        ref_props = ref_data.value_counts(normalize=True)
        curr_props = curr_data.value_counts(normalize=True)
        
        all_categories = set(ref_props.index) | set(curr_props.index)
        
        psi = 0.0
        for cat in all_categories:
            ref_prop = ref_props.get(cat, epsilon)
            curr_prop = curr_props.get(cat, epsilon)
            psi += (curr_prop - ref_prop) * np.log(curr_prop / ref_prop)
        
        return abs(psi)
    
    def _calculate_jsd(self, ref_data: pd.Series, curr_data: pd.Series, 
                      bins: int = 10) -> float:
        """
        計算 Jensen-Shannon Divergence
        """
        try:
            breakpoints = np.percentile(ref_data, np.linspace(0, 100, bins + 1))
            breakpoints = np.unique(breakpoints)
            
            if len(breakpoints) < 2:
                return 0.0
            
            ref_hist = np.histogram(ref_data, bins=breakpoints)[0]
            curr_hist = np.histogram(curr_data, bins=breakpoints)[0]
            
            # 正規化為機率分布
            ref_hist = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
            curr_hist = curr_hist / curr_hist.sum() if curr_hist.sum() > 0 else curr_hist
            
            # 計算 JSD
            jsd = jensenshannon(ref_hist, curr_hist, base=2)
            
            return float(jsd)
        except:
            return 0.0
    
    def _calculate_tvd(self, ref_data: pd.Series, curr_data: pd.Series) -> float:
        """
        計算 Total Variation Distance
        TVD = 0.5 * Σ|P(x) - Q(x)|
        """
        ref_props = ref_data.value_counts(normalize=True)
        curr_props = curr_data.value_counts(normalize=True)
        
        all_categories = set(ref_props.index) | set(curr_props.index)
        
        tvd = 0.0
        for cat in all_categories:
            ref_prop = ref_props.get(cat, 0.0)
            curr_prop = curr_props.get(cat, 0.0)
            tvd += abs(curr_prop - ref_prop)
        
        return 0.5 * tvd
    
    def _calculate_hellinger(self, ref_data: pd.Series, curr_data: pd.Series) -> float:
        """
        計算 Hellinger Distance
        H(P,Q) = sqrt(1 - Σ sqrt(P(x) * Q(x)))
        """
        ref_props = ref_data.value_counts(normalize=True)
        curr_props = curr_data.value_counts(normalize=True)
        
        all_categories = set(ref_props.index) | set(curr_props.index)
        
        bc_coef = 0.0  # Bhattacharyya coefficient
        for cat in all_categories:
            ref_prop = ref_props.get(cat, 0.0)
            curr_prop = curr_props.get(cat, 0.0)
            bc_coef += np.sqrt(ref_prop * curr_prop)
        
        hellinger = np.sqrt(1 - bc_coef)
        
        return hellinger
    
    # 解釋函數
    def _interpret_psi(self, psi_value: float) -> str:
        """解釋 PSI 值"""
        if psi_value < 0.1:
            return "無顯著變化"
        elif psi_value < 0.2:
            return "輕微變化"
        else:
            return "顯著漂移"
    
    def _interpret_jsd(self, jsd_value: float) -> str:
        """解釋 JSD 值"""
        if jsd_value < 0.05:
            return "幾乎相同"
        elif jsd_value < 0.1:
            return "輕微差異"
        elif jsd_value < 0.2:
            return "中度差異"
        else:
            return "顯著差異"
    
    def _interpret_tvd(self, tvd_value: float) -> str:
        """解釋 TVD 值"""
        if tvd_value < 0.1:
            return "非常相似"
        elif tvd_value < 0.2:
            return "輕微差異"
        elif tvd_value < 0.3:
            return "中度差異"
        else:
            return "顯著差異"
    
    def _interpret_hellinger(self, hellinger_value: float) -> str:
        """解釋 Hellinger 值"""
        if hellinger_value < 0.1:
            return "非常相似"
        elif hellinger_value < 0.2:
            return "輕微差異"
        elif hellinger_value < 0.3:
            return "中度差異"
        else:
            return "顯著差異"
    
    def _interpret_wasserstein(self, wass_dist: float, 
                              ref_data: pd.Series, 
                              curr_data: pd.Series) -> str:
        """
        解釋 Wasserstein Distance
        相對於資料範圍來評估
        """
        data_range = max(ref_data.max(), curr_data.max()) - min(ref_data.min(), curr_data.min())
        if data_range > 0:
            relative_dist = wass_dist / data_range
            if relative_dist < 0.05:
                return "非常相似"
            elif relative_dist < 0.1:
                return "輕微差異"
            elif relative_dist < 0.2:
                return "中度差異"
            else:
                return "顯著差異"
        else:
            return "無法評估"
    
    def generate_report(self, output_format: str = 'text') -> str:
        """
        生成漂移偵測報告
        """
        if not self.drift_results:
            return "尚未執行漂移偵測,請先呼叫 detect_drift() 方法"
        
        if output_format == 'markdown':
            return self._generate_markdown_report()
        else:
            return self._generate_text_report()
    
    def _generate_text_report(self) -> str:
        """生成文字格式報告"""
        report = []
        report.append("=" * 70)
        report.append("資料漂移偵測報告 (改進版)")
        report.append("=" * 70)
        report.append("")
        
        summary = self.drift_results['summary']
        report.append(f"📊 總特徵數: {summary['total_features']}")
        report.append(f"⚠️  發生漂移的特徵數: {summary['drifted_features']}")
        report.append(f"📈 漂移比例: {summary['drift_percentage']:.2f}%")
        
        if summary['warnings']:
            report.append(f"\n⚠️  警告訊息:")
            for warning in summary['warnings']:
                report.append(f"   - {warning}")
        
        report.append("")
        report.append("-" * 70)
        
        for feature, result in self.drift_results['feature_results'].items():
            report.append("")
            report.append(f"🔍 特徵: {feature}")
            report.append(f"   類型: {result['feature_type']}")
            report.append(f"   漂移狀態: {'❌ 有漂移' if result['has_drift'] else '✅ 無漂移'}")
            
            if result['has_drift']:
                report.append(f"   漂移原因: {', '.join(result.get('drift_reasons', []))}")
            
            # 樣本檢查
            sample_check = result.get('sample_check', {})
            if not sample_check.get('is_valid', True):
                report.append(f"   ⚠️  樣本警告:")
                for warning in sample_check.get('warnings', []):
                    report.append(f"      - {warning}")
            
            # 缺失值分析
            missing = result.get('missing_analysis', {})
            if missing:
                report.append(f"   缺失值比例: {missing['reference_missing_rate']:.2%} → {missing['current_missing_rate']:.2%}")
                if missing['has_drift']:
                    report.append(f"   ⚠️  缺失值有顯著變化 (p={missing['p_value']:.4f})")
            
            # 詳細測試結果
            if result['feature_type'] == 'numerical':
                report.append("   \n   統計檢定:")
                for test_name, test_result in result['tests'].items():
                    if test_name == 'ks_test':
                        report.append(f"      • KS Test: p={test_result['p_value']:.4f}")
                    elif test_name == 'psi':
                        report.append(f"      • PSI: {test_result['value']:.4f} ({test_result['interpretation']})")
                    elif test_name == 'jensen_shannon_divergence':
                        report.append(f"      • JSD: {test_result['value']:.4f} ({test_result['interpretation']})")
                    elif test_name == 'mann_whitney':
                        report.append(f"      • Mann-Whitney: p={test_result['p_value']:.4f}")
            else:
                report.append("   \n   統計檢定:")
                for test_name, test_result in result['tests'].items():
                    if test_name == 'chi_square_test':
                        report.append(f"      • Chi-Square: p={test_result['p_value']:.4f}")
                    elif test_name == 'psi':
                        report.append(f"      • PSI: {test_result['value']:.4f} ({test_result['interpretation']})")
                    elif test_name == 'total_variation_distance':
                        report.append(f"      • TVD: {test_result['value']:.4f} ({test_result['interpretation']})")
                    elif test_name == 'hellinger_distance':
                        report.append(f"      • Hellinger: {test_result['value']:.4f} ({test_result['interpretation']})")
            
            report.append("-" * 70)
        
        return "\n".join(report)
    
    def _generate_markdown_report(self) -> str:
        """生成 Markdown 格式報告"""
        report = []
        report.append("# 資料漂移偵測報告 (改進版)")
        report.append("")
        
        summary = self.drift_results['summary']
        report.append("## 📊 摘要")
        report.append(f"- **總特徵數**: {summary['total_features']}")
        report.append(f"- **發生漂移的特徵數**: {summary['drifted_features']}")
        report.append(f"- **漂移比例**: {summary['drift_percentage']:.2f}%")
        
        if summary['warnings']:
            report.append(f"\n### ⚠️ 警告訊息")
            for warning in summary['warnings']:
                report.append(f"- {warning}")
        
        report.append("")
        report.append("## 🔍 詳細結果")
        
        for feature, result in self.drift_results['feature_results'].items():
            status = "❌ 有漂移" if result['has_drift'] else "✅ 無漂移"
            report.append(f"\n### {feature} - {status}")
            report.append(f"**類型**: {result['feature_type']}")
            
            if result['has_drift']:
                report.append(f"**漂移原因**: {', '.join(result.get('drift_reasons', []))}")
            
            report.append("")
            
            # 缺失值分析
            missing = result.get('missing_analysis', {})
            if missing and missing.get('reference_missing_rate', 0) > 0:
                report.append(f"**缺失值**: {missing['reference_missing_rate']:.2%} → {missing['current_missing_rate']:.2%}")
                report.append("")
            
            # 詳細測試結果
            report.append("#### 統計檢定結果")
            for test_name, test_result in result['tests'].items():
                if 'value' in test_result:
                    report.append(f"- **{test_name}**: {test_result['value']:.4f} - {test_result.get('interpretation', '')}")
                elif 'p_value' in test_result:
                    report.append(f"- **{test_name}**: p-value = {test_result['p_value']:.4f}")
            
            report.append("")
        
        return "\n".join(report)
    
    def get_drifted_features(self) -> List[str]:
        """取得所有發生漂移的特徵名稱"""
        if not self.drift_results:
            return []
        
        drifted = []
        for feature, result in self.drift_results['feature_results'].items():
            if result['has_drift']:
                drifted.append(feature)
        
        return drifted


if __name__ == "__main__":
    print("改進版資料漂移偵測系統已就緒")
    print("\n新增功能:")
    print("✅ Jensen-Shannon Divergence")
    print("✅ Mann-Whitney U Test")
    print("✅ Anderson-Darling Test (參考)")
    print("✅ Total Variation Distance")
    print("✅ Hellinger Distance")
    print("✅ 缺失值漂移檢測")
    print("✅ 樣本大小檢查")
    print("✅ 新類別檢測")
    print("✅ 改進的 PSI 計算")
