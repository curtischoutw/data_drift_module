"""
資料漂移視覺化模組
提供完整的圖表來展示資料漂移情況
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DriftVisualizer:
    """資料漂移視覺化工具"""

    def __init__(self, detector, current_data: pd.DataFrame):
        """
        初始化視覺化工具

        Parameters:
        -----------
        detector : DataDriftDetector
            已執行漂移偵測的偵測器實例
        current_data : pd.DataFrame
            當前資料集
        """
        self.detector = detector
        self.drift_results = detector.drift_results
        self.reference_data = detector.reference_data
        self.current_data = current_data

    def plot_drift_summary(self, figsize=(14, 6), save_path=None):
        """繪製漂移摘要圖（圓餅圖 + 特徵狀態 + 漂移嚴重程度）"""
        if not self.drift_results:
            print("尚未執行漂移偵測")
            return

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        summary = self.drift_results['summary']
        drifted = summary['drifted_features']
        no_drift = summary['total_features'] - drifted

        # 左圖：漂移比例圓餅圖
        axes[0].pie([drifted, no_drift],
                    labels=['有漂移', '無漂移'],
                    autopct='%1.1f%%',
                    colors=['#ff6b6b', '#51cf66'],
                    explode=(0.1, 0),
                    startangle=90)
        axes[0].set_title('特徵漂移比例', fontsize=14, fontweight='bold')

        # 中圖：各特徵漂移狀態
        feature_names = []
        has_drift_values = []
        drift_reasons_count = []

        for feature, result in self.drift_results['feature_results'].items():
            feature_names.append(feature)
            has_drift_values.append(1 if result['has_drift'] else 0)
            drift_reasons_count.append(len(result.get('drift_reasons', [])))

        colors_bar = ['#ff6b6b' if x == 1 else '#51cf66' for x in has_drift_values]
        y_pos = np.arange(len(feature_names))

        axes[1].barh(y_pos, has_drift_values, color=colors_bar, alpha=0.7)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(feature_names)
        axes[1].set_xlabel('漂移狀態 (0=無, 1=有)')
        axes[1].set_title('各特徵漂移狀態', fontsize=14, fontweight='bold')
        axes[1].set_xlim([0, 1.2])
        axes[1].grid(True, alpha=0.3, axis='x')

        # 右圖：漂移嚴重程度（漂移原因數量）
        axes[2].barh(y_pos, drift_reasons_count,
                     color=['#ff6b6b' if c > 0 else '#51cf66' for c in drift_reasons_count],
                     alpha=0.7)
        axes[2].set_yticks(y_pos)
        axes[2].set_yticklabels(feature_names)
        axes[2].set_xlabel('檢測到的漂移原因數量')
        axes[2].set_title('漂移嚴重程度', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已儲存至: {save_path}")

        plt.show()

    def plot_numerical_drift(self, feature_name: str, figsize=(16, 10), save_path=None):
        """
        繪製數值型特徵的完整漂移分析圖

        Parameters:
        -----------
        feature_name : str
            特徵名稱
        """
        if feature_name not in self.drift_results['feature_results']:
            print(f"特徵 '{feature_name}' 不存在")
            return

        result = self.drift_results['feature_results'][feature_name]

        if result['feature_type'] != 'numerical':
            print(f"特徵 '{feature_name}' 不是數值型變數")
            return

        ref_data = self.reference_data[feature_name].dropna()
        curr_data = self.current_data[feature_name].dropna()

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 分布比較（直方圖）
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.hist(ref_data, bins=30, alpha=0.6, label='參考資料',
                 color='blue', density=True, edgecolor='black')
        ax1.hist(curr_data, bins=30, alpha=0.6, label='當前資料',
                 color='red', density=True, edgecolor='black')
        ax1.set_xlabel('值', fontsize=11)
        ax1.set_ylabel('密度', fontsize=11)
        ax1.set_title(f'{feature_name} - 分布比較', fontweight='bold', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 箱型圖
        ax2 = fig.add_subplot(gs[0, 2])
        bp = ax2.boxplot([ref_data, curr_data],
                         labels=['參考', '當前'],
                         patch_artist=True,
                         showmeans=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax2.set_ylabel('值', fontsize=11)
        ax2.set_title('箱型圖', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Q-Q Plot
        ax3 = fig.add_subplot(gs[1, 0])
        quantiles = np.linspace(0.01, 0.99, 100)
        ref_quantiles = np.quantile(ref_data, quantiles)
        curr_quantiles = np.quantile(curr_data, quantiles)
        ax3.scatter(ref_quantiles, curr_quantiles, alpha=0.5, s=20)
        min_val = min(ref_quantiles.min(), curr_quantiles.min())
        max_val = max(ref_quantiles.max(), curr_quantiles.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
        ax3.set_xlabel('參考資料分位數', fontsize=10)
        ax3.set_ylabel('當前資料分位數', fontsize=10)
        ax3.set_title('Q-Q Plot', fontweight='bold', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. CDF 比較
        ax4 = fig.add_subplot(gs[1, 1])
        ref_sorted = np.sort(ref_data)
        curr_sorted = np.sort(curr_data)
        ref_cdf = np.arange(1, len(ref_sorted) + 1) / len(ref_sorted)
        curr_cdf = np.arange(1, len(curr_sorted) + 1) / len(curr_sorted)
        ax4.plot(ref_sorted, ref_cdf, label='參考資料', color='blue', linewidth=2)
        ax4.plot(curr_sorted, curr_cdf, label='當前資料', color='red', linewidth=2)
        ax4.set_xlabel('值', fontsize=10)
        ax4.set_ylabel('累積機率', fontsize=10)
        ax4.set_title('累積分布函數 (CDF)', fontweight='bold', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. 統計量比較
        ax5 = fig.add_subplot(gs[1, 2])
        ref_stats = result['statistics']['reference']
        curr_stats = result['statistics']['current']

        metrics = ['平均值', '標準差', '中位數', '偏度', '峰度']
        ref_values = [
            ref_stats['mean'],
            ref_stats['std'],
            ref_stats['median'],
            ref_stats.get('skewness', 0),
            ref_stats.get('kurtosis', 0)
        ]
        curr_values = [
            curr_stats['mean'],
            curr_stats['std'],
            curr_stats['median'],
            curr_stats.get('skewness', 0),
            curr_stats.get('kurtosis', 0)
        ]

        x = np.arange(len(metrics))
        width = 0.35
        ax5.barh(x - width/2, ref_values, width, label='參考', color='blue', alpha=0.7)
        ax5.barh(x + width/2, curr_values, width, label='當前', color='red', alpha=0.7)
        ax5.set_yticks(x)
        ax5.set_yticklabels(metrics, fontsize=10)
        ax5.set_xlabel('值', fontsize=10)
        ax5.set_title('統計量比較', fontweight='bold', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. 測試結果摘要
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')

        tests_info = ["【統計檢定結果】", ""]
        for test_name, test_result in result['tests'].items():
            if test_name == 'ks_test':
                tests_info.append(f"✓ KS Test: p-value = {test_result['p_value']:.4f} ({test_result['interpretation']})")
            elif test_name == 'psi':
                tests_info.append(f"✓ PSI: {test_result['value']:.4f} ({test_result['interpretation']})")
            elif test_name == 'jensen_shannon_divergence':
                tests_info.append(f"✓ Jensen-Shannon Divergence: {test_result['value']:.4f} ({test_result['interpretation']})")
            elif test_name == 'wasserstein_distance':
                tests_info.append(f"  Wasserstein Distance: {test_result['value']:.4f} ({test_result['interpretation']}) [輔助]")
            elif test_name == 'mann_whitney':
                tests_info.append(f"✓ Mann-Whitney U: p-value = {test_result['p_value']:.4f} ({test_result['interpretation']})")
            elif test_name == 'anderson_darling':
                tests_info.append(f"  Anderson-Darling: statistic = {test_result['statistic']:.4f} [輔助]")

        missing = result.get('missing_analysis', {})
        if missing:
            tests_info.extend([
                "",
                "【缺失值分析】",
                f"參考資料缺失率: {missing['reference_missing_rate']:.2%}",
                f"當前資料缺失率: {missing['current_missing_rate']:.2%}"
            ])
            if missing['has_drift']:
                tests_info.append("⚠️ 缺失率有顯著變化!")

        ax6.text(0.05, 0.95, '\n'.join(tests_info), transform=ax6.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                 family='monospace')

        drift_status = "偵測到漂移" if result['has_drift'] else "無漂移"
        fig.suptitle(f'特徵: {feature_name} ({drift_status})',
                     fontsize=16, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已儲存至: {save_path}")

        plt.show()

    def plot_categorical_drift(self, feature_name: str, figsize=(14, 10), save_path=None):
        """
        繪製類別型特徵的完整漂移分析圖

        Parameters:
        -----------
        feature_name : str
            特徵名稱
        """
        if feature_name not in self.drift_results['feature_results']:
            print(f"特徵 '{feature_name}' 不存在")
            return

        result = self.drift_results['feature_results'][feature_name]

        if result['feature_type'] != 'categorical':
            print(f"特徵 '{feature_name}' 不是類別型變數")
            return

        ref_dist = result['statistics']['reference_distribution']
        curr_dist = result['statistics']['current_distribution']

        all_categories = sorted(set(ref_dist.keys()) | set(curr_dist.keys()))
        ref_values = [ref_dist.get(cat, 0) for cat in all_categories]
        curr_values = [curr_dist.get(cat, 0) for cat in all_categories]

        ref_total = sum(ref_values)
        curr_total = sum(curr_values)
        ref_props = [v / ref_total if ref_total > 0 else 0 for v in ref_values]
        curr_props = [v / curr_total if curr_total > 0 else 0 for v in curr_values]

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. 條形圖比較
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(len(all_categories))
        width = 0.35
        ax1.bar(x - width/2, ref_props, width, label='參考資料', color='blue', alpha=0.7)
        ax1.bar(x + width/2, curr_props, width, label='當前資料', color='red', alpha=0.7)
        ax1.set_xlabel('類別', fontsize=11)
        ax1.set_ylabel('比例', fontsize=11)
        ax1.set_title(f'{feature_name} - 類別分布比較', fontweight='bold', fontsize=13)
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_categories, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. 差異橫條圖
        ax2 = fig.add_subplot(gs[1, 0])
        diff = [curr_props[i] - ref_props[i] for i in range(len(all_categories))]
        ax2.barh(all_categories, diff,
                 color=['red' if d > 0 else 'blue' for d in diff], alpha=0.7)
        ax2.set_xlabel('比例差異 (當前 - 參考)', fontsize=11)
        ax2.set_title('分布變化', fontweight='bold', fontsize=12)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. 參考資料圓餅圖
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.pie(ref_props, labels=all_categories, autopct='%1.1f%%', startangle=90)
        ax3.set_title('參考資料分布', fontweight='bold', fontsize=12)

        # 4. 當前資料圓餅圖
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.pie(curr_props, labels=all_categories, autopct='%1.1f%%', startangle=90)
        ax4.set_title('當前資料分布', fontweight='bold', fontsize=12)

        # 5. 測試結果摘要
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        tests_info = ["【統計檢定結果】", ""]
        for test_name, test_result in result['tests'].items():
            if test_name == 'chi_square_test':
                tests_info.append(f"✓ Chi-Square: p = {test_result['p_value']:.4f}")
                tests_info.append(f"  ({test_result['interpretation']})")
            elif test_name == 'psi':
                tests_info.append(f"✓ PSI: {test_result['value']:.4f}")
                tests_info.append(f"  ({test_result['interpretation']})")

        stats = result['statistics']
        new_cats = stats.get('new_categories', [])
        disappeared_cats = stats.get('disappeared_categories', [])

        if new_cats or disappeared_cats:
            tests_info.extend(["", "【類別變化】"])
            if new_cats:
                tests_info.append(f"新出現類別: {new_cats}")
            if disappeared_cats:
                tests_info.append(f"消失類別: {disappeared_cats}")

        ax5.text(0.05, 0.95, '\n'.join(tests_info), transform=ax5.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                 family='monospace')

        drift_status = "偵測到漂移" if result['has_drift'] else "無漂移"
        fig.suptitle(f'特徵: {feature_name} ({drift_status})',
                     fontsize=16, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已儲存至: {save_path}")

        plt.show()

    def plot_metrics_comparison(self, figsize=(14, 8), save_path=None):
        """繪製所有特徵的多指標比較圖（PSI 值、漂移原因、類型分析、缺失值）"""
        if not self.drift_results:
            print("尚未執行漂移偵測")
            return

        feature_names = []
        psi_values = []

        for feature, result in self.drift_results['feature_results'].items():
            feature_names.append(feature)
            psi_values.append(result['tests']['psi']['value'])

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. PSI 值比較
        colors = ['#ff6b6b' if psi > 0.2 else '#ffd93d' if psi > 0.1 else '#51cf66'
                  for psi in psi_values]
        axes[0, 0].barh(feature_names, psi_values, color=colors)
        axes[0, 0].axvline(x=0.1, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        axes[0, 0].axvline(x=0.2, color='red', linestyle='--', linewidth=1, alpha=0.7)
        axes[0, 0].set_xlabel('PSI 值', fontweight='bold')
        axes[0, 0].set_title('各特徵 PSI 值', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='x')

        # 2. 漂移原因統計
        drift_reasons_all = {}
        for feature, result in self.drift_results['feature_results'].items():
            for reason in result.get('drift_reasons', []):
                drift_reasons_all[reason] = drift_reasons_all.get(reason, 0) + 1

        if drift_reasons_all:
            reasons = list(drift_reasons_all.keys())
            counts = list(drift_reasons_all.values())
            axes[0, 1].bar(reasons, counts, color='coral', alpha=0.7)
            axes[0, 1].set_xlabel('漂移原因', fontweight='bold')
            axes[0, 1].set_ylabel('出現次數', fontweight='bold')
            axes[0, 1].set_title('漂移原因分布', fontsize=13, fontweight='bold')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        else:
            axes[0, 1].text(0.5, 0.5, '無漂移檢測', ha='center', va='center', fontsize=14)
            axes[0, 1].axis('off')

        # 3. 數值型 vs 類別型漂移比例
        numerical_drifted = categorical_drifted = 0
        numerical_total = categorical_total = 0

        for feature, result in self.drift_results['feature_results'].items():
            if result['feature_type'] == 'numerical':
                numerical_total += 1
                if result['has_drift']:
                    numerical_drifted += 1
            else:
                categorical_total += 1
                if result['has_drift']:
                    categorical_drifted += 1

        categories = ['數值型', '類別型']
        drifted_counts = [numerical_drifted, categorical_drifted]
        total_counts = [numerical_total, categorical_total]
        no_drift_counts = [total_counts[i] - drifted_counts[i] for i in range(2)]

        x = np.arange(len(categories))
        width = 0.35
        axes[1, 0].bar(x, drifted_counts, width, label='有漂移', color='#ff6b6b', alpha=0.7)
        axes[1, 0].bar(x, no_drift_counts, width, bottom=drifted_counts,
                       label='無漂移', color='#51cf66', alpha=0.7)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(categories)
        axes[1, 0].set_ylabel('特徵數量', fontweight='bold')
        axes[1, 0].set_title('按類型分析漂移', fontsize=13, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 4. 缺失值變化
        missing_changes = []
        features_with_missing = []

        for feature, result in self.drift_results['feature_results'].items():
            missing = result.get('missing_analysis', {})
            if missing and missing.get('reference_missing_rate', 0) > 0:
                missing_change = abs(
                    missing['current_missing_rate'] -
                    missing['reference_missing_rate']
                )
                if missing_change > 0.01:
                    missing_changes.append(missing_change)
                    features_with_missing.append(feature)

        if features_with_missing:
            axes[1, 1].barh(features_with_missing, missing_changes, color='#ffd93d', alpha=0.7)
            axes[1, 1].set_xlabel('缺失率變化', fontweight='bold')
            axes[1, 1].set_title('缺失值比例變化', fontsize=13, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3, axis='x')
        else:
            axes[1, 1].text(0.5, 0.5, '無顯著缺失值變化', ha='center', va='center', fontsize=14)
            axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已儲存至: {save_path}")

        plt.show()

    def plot_all_drifts(self, save_dir=None):
        """
        繪製所有特徵的漂移分析圖

        Parameters:
        -----------
        save_dir : str
            儲存目錄路徑
        """
        if not self.drift_results:
            print("尚未執行漂移偵測")
            return

        self.plot_drift_summary(
            save_path=f"{save_dir}/drift_summary.png" if save_dir else None
        )
        self.plot_metrics_comparison(
            save_path=f"{save_dir}/metrics_comparison.png" if save_dir else None
        )

        for feature, result in self.drift_results['feature_results'].items():
            feature_path = f"{save_dir}/{feature}_drift.png" if save_dir else None

            if result['feature_type'] == 'numerical':
                self.plot_numerical_drift(feature, save_path=feature_path)
            else:
                self.plot_categorical_drift(feature, save_path=feature_path)

        print("所有圖表繪製完成!")


if __name__ == "__main__":
    print("資料漂移視覺化模組已就緒")
