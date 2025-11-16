"""
資料漂移偵測系統 (Data Drift Detection System)
支援多種統計檢定方法和視覺化功能
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataDriftDetector:
    """
    資料漂移偵測器
    支援連續型和類別型變數的漂移偵測
    """
    
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        """
        初始化偵測器
        
        Parameters:
        -----------
        reference_data : pd.DataFrame
            基準資料集（通常是訓練資料或穩定期的生產資料）
        threshold : float
            顯著性水準（預設 0.05）
        """
        self.reference_data = reference_data
        self.threshold = threshold
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
                'drift_percentage': 0.0
            },
            'feature_results': {}
        }
        
        for column in self.reference_data.columns:
            if column not in current_data.columns:
                print(f"警告: 欄位 '{column}' 不存在於當前資料中")
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
        # 移除缺失值
        ref_clean = ref_data.dropna()
        curr_clean = curr_data.dropna()
        
        # 判斷資料類型
        if self._is_numerical(ref_clean):
            return self._detect_numerical_drift(column_name, ref_clean, curr_clean)
        else:
            return self._detect_categorical_drift(column_name, ref_clean, curr_clean)
    
    def _is_numerical(self, data: pd.Series) -> bool:
        """判斷是否為數值型資料"""
        return pd.api.types.is_numeric_dtype(data)
    
    def _detect_numerical_drift(self, column_name: str,
                               ref_data: pd.Series, 
                               curr_data: pd.Series) -> Dict:
        """
        偵測連續型變數的漂移
        使用 KS 檢定、Wasserstein Distance 和 PSI
        """
        # KS 檢定
        ks_statistic, ks_pvalue = stats.ks_2samp(ref_data, curr_data)
        
        # Wasserstein Distance
        wasserstein_dist = stats.wasserstein_distance(ref_data, curr_data)
        
        # PSI (Population Stability Index)
        psi_value = self._calculate_psi(ref_data, curr_data)
        
        # 統計摘要
        stats_summary = {
            'reference': {
                'mean': float(ref_data.mean()),
                'std': float(ref_data.std()),
                'median': float(ref_data.median()),
                'min': float(ref_data.min()),
                'max': float(ref_data.max())
            },
            'current': {
                'mean': float(curr_data.mean()),
                'std': float(curr_data.std()),
                'median': float(curr_data.median()),
                'min': float(curr_data.min()),
                'max': float(curr_data.max())
            }
        }
        
        return {
            'feature_name': column_name,
            'feature_type': 'numerical',
            'has_drift': ks_pvalue < self.threshold or psi_value > 0.2,
            'tests': {
                'ks_test': {
                    'statistic': float(ks_statistic),
                    'p_value': float(ks_pvalue),
                    'interpretation': 'Drift detected' if ks_pvalue < self.threshold else 'No drift'
                },
                'wasserstein_distance': {
                    'value': float(wasserstein_dist),
                    'interpretation': 'High distance' if wasserstein_dist > 1.0 else 'Low distance'
                },
                'psi': {
                    'value': float(psi_value),
                    'interpretation': self._interpret_psi(psi_value)
                }
            },
            'statistics': stats_summary
        }
    
    def _detect_categorical_drift(self, column_name: str,
                                  ref_data: pd.Series, 
                                  curr_data: pd.Series) -> Dict:
        """
        偵測類別型變數的漂移
        使用 Chi-Square 檢定和 PSI
        """
        # 取得所有類別
        all_categories = sorted(set(ref_data.unique()) | set(curr_data.unique()))
        
        # 計算頻率
        ref_counts = ref_data.value_counts()
        curr_counts = curr_data.value_counts()
        
        # 對齊類別並正規化為相同總數
        ref_freq = []
        curr_freq = []
        for cat in all_categories:
            ref_freq.append(ref_counts.get(cat, 0))
            curr_freq.append(curr_counts.get(cat, 0))
        
        # 轉換為比例，然後乘以相同的樣本數以確保總和相等
        ref_total = sum(ref_freq)
        curr_total = sum(curr_freq)
        
        # 使用較小的樣本數作為基準
        base_n = min(ref_total, curr_total)
        
        ref_freq_normalized = [int(f / ref_total * base_n) if f > 0 else 0 for f in ref_freq]
        curr_freq_normalized = [int(f / curr_total * base_n) if f > 0 else 0 for f in curr_freq]
        
        # 確保至少有1個觀測值以避免除零
        ref_freq_normalized = [max(1, f) for f in ref_freq_normalized]
        curr_freq_normalized = [max(1, f) for f in curr_freq_normalized]
        
        # Chi-Square 檢定
        try:
            chi2_stat, chi2_pvalue = stats.chisquare(
                f_obs=curr_freq_normalized, 
                f_exp=ref_freq_normalized
            )
        except:
            # 如果還是失敗，使用替代方法
            chi2_stat = 0.0
            chi2_pvalue = 1.0
        
        # PSI for categorical
        psi_value = self._calculate_categorical_psi(ref_data, curr_data)
        
        return {
            'feature_name': column_name,
            'feature_type': 'categorical',
            'has_drift': chi2_pvalue < self.threshold or psi_value > 0.2,
            'tests': {
                'chi_square_test': {
                    'statistic': float(chi2_stat),
                    'p_value': float(chi2_pvalue),
                    'interpretation': 'Drift detected' if chi2_pvalue < self.threshold else 'No drift'
                },
                'psi': {
                    'value': float(psi_value),
                    'interpretation': self._interpret_psi(psi_value)
                }
            },
            'statistics': {
                'reference_categories': len(ref_data.unique()),
                'current_categories': len(curr_data.unique()),
                'reference_distribution': ref_data.value_counts().to_dict(),
                'current_distribution': curr_data.value_counts().to_dict()
            }
        }
    
    def _calculate_psi(self, ref_data: pd.Series, curr_data: pd.Series, 
                       bins: int = 10) -> float:
        """
        計算 Population Stability Index (PSI)
        適用於連續型變數
        """
        # 使用參考資料的分位數建立分箱
        try:
            breakpoints = np.percentile(ref_data, np.linspace(0, 100, bins + 1))
            breakpoints = np.unique(breakpoints)  # 移除重複值
            
            if len(breakpoints) < 2:
                return 0.0
                
            ref_counts = np.histogram(ref_data, bins=breakpoints)[0]
            curr_counts = np.histogram(curr_data, bins=breakpoints)[0]
            
            # 計算比例並避免除以零
            ref_props = (ref_counts + 1) / (len(ref_data) + bins)
            curr_props = (curr_counts + 1) / (len(curr_data) + bins)
            
            # 計算 PSI
            psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
            
            return abs(psi)
        except:
            return 0.0
    
    def _calculate_categorical_psi(self, ref_data: pd.Series, 
                                   curr_data: pd.Series) -> float:
        """
        計算類別型變數的 PSI
        """
        ref_props = ref_data.value_counts(normalize=True)
        curr_props = curr_data.value_counts(normalize=True)
        
        all_categories = set(ref_props.index) | set(curr_props.index)
        
        psi = 0.0
        for cat in all_categories:
            ref_prop = ref_props.get(cat, 0.0001)
            curr_prop = curr_props.get(cat, 0.0001)
            psi += (curr_prop - ref_prop) * np.log(curr_prop / ref_prop)
        
        return abs(psi)
    
    def _interpret_psi(self, psi_value: float) -> str:
        """解釋 PSI 值"""
        if psi_value < 0.1:
            return "無顯著變化 (No significant change)"
        elif psi_value < 0.2:
            return "輕微變化 (Slight change)"
        else:
            return "顯著漂移 (Significant drift)"
    
    def generate_report(self, output_format: str = 'text') -> str:
        """
        生成漂移偵測報告
        
        Parameters:
        -----------
        output_format : str
            輸出格式 ('text' 或 'markdown')
        """
        if not self.drift_results:
            return "尚未執行漂移偵測，請先呼叫 detect_drift() 方法"
        
        if output_format == 'markdown':
            return self._generate_markdown_report()
        else:
            return self._generate_text_report()
    
    def _generate_text_report(self) -> str:
        """生成文字格式報告"""
        report = []
        report.append("=" * 60)
        report.append("資料漂移偵測報告 (Data Drift Detection Report)")
        report.append("=" * 60)
        report.append("")
        
        summary = self.drift_results['summary']
        report.append(f"總特徵數: {summary['total_features']}")
        report.append(f"發生漂移的特徵數: {summary['drifted_features']}")
        report.append(f"漂移比例: {summary['drift_percentage']:.2f}%")
        report.append("")
        report.append("-" * 60)
        
        for feature, result in self.drift_results['feature_results'].items():
            report.append("")
            report.append(f"特徵: {feature}")
            report.append(f"類型: {result['feature_type']}")
            report.append(f"漂移狀態: {'⚠️  有漂移' if result['has_drift'] else '✓ 無漂移'}")
            
            if result['feature_type'] == 'numerical':
                ks = result['tests']['ks_test']
                psi = result['tests']['psi']
                report.append(f"  KS 檢定 p-value: {ks['p_value']:.4f} - {ks['interpretation']}")
                report.append(f"  PSI 值: {psi['value']:.4f} - {psi['interpretation']}")
                
                ref_stats = result['statistics']['reference']
                curr_stats = result['statistics']['current']
                report.append(f"  平均值變化: {ref_stats['mean']:.4f} → {curr_stats['mean']:.4f}")
                report.append(f"  標準差變化: {ref_stats['std']:.4f} → {curr_stats['std']:.4f}")
            else:
                chi2 = result['tests']['chi_square_test']
                psi = result['tests']['psi']
                report.append(f"  Chi-Square p-value: {chi2['p_value']:.4f} - {chi2['interpretation']}")
                report.append(f"  PSI 值: {psi['value']:.4f} - {psi['interpretation']}")
            
            report.append("-" * 60)
        
        return "\n".join(report)
    
    def _generate_markdown_report(self) -> str:
        """生成 Markdown 格式報告"""
        report = []
        report.append("# 資料漂移偵測報告")
        report.append("")
        
        summary = self.drift_results['summary']
        report.append("## 摘要")
        report.append(f"- **總特徵數**: {summary['total_features']}")
        report.append(f"- **發生漂移的特徵數**: {summary['drifted_features']}")
        report.append(f"- **漂移比例**: {summary['drift_percentage']:.2f}%")
        report.append("")
        
        report.append("## 詳細結果")
        
        for feature, result in self.drift_results['feature_results'].items():
            status = "⚠️ 有漂移" if result['has_drift'] else "✓ 無漂移"
            report.append(f"### {feature} - {status}")
            report.append(f"**類型**: {result['feature_type']}")
            report.append("")
            
            if result['feature_type'] == 'numerical':
                report.append("#### 統計檢定結果")
                ks = result['tests']['ks_test']
                psi = result['tests']['psi']
                report.append(f"- **KS 檢定**: p-value = {ks['p_value']:.4f} ({ks['interpretation']})")
                report.append(f"- **PSI**: {psi['value']:.4f} ({psi['interpretation']})")
                
                report.append("")
                report.append("#### 統計量變化")
                ref = result['statistics']['reference']
                curr = result['statistics']['current']
                report.append(f"- **平均值**: {ref['mean']:.4f} → {curr['mean']:.4f}")
                report.append(f"- **標準差**: {ref['std']:.4f} → {curr['std']:.4f}")
            else:
                report.append("#### 統計檢定結果")
                chi2 = result['tests']['chi_square_test']
                psi = result['tests']['psi']
                report.append(f"- **Chi-Square 檢定**: p-value = {chi2['p_value']:.4f} ({chi2['interpretation']})")
                report.append(f"- **PSI**: {psi['value']:.4f} ({psi['interpretation']})")
            
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
    # 使用範例
    print("資料漂移偵測系統已就緒")
    print("請參考 example_usage.py 了解如何使用")
