"""
資料漂移視覺化模組
提供各種圖表來展示資料漂移情況
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DriftVisualizer:
    """資料漂移視覺化工具"""
    
    def __init__(self, detector):
        """
        初始化視覺化工具
        
        Parameters:
        -----------
        detector : DataDriftDetector
            已執行漂移偵測的偵測器實例
        """
        self.detector = detector
        self.drift_results = detector.drift_results
        
    def plot_drift_summary(self, figsize=(12, 6), save_path=None):
        """
        繪製漂移摘要圖
        """
        if not self.drift_results:
            print("尚未執行漂移偵測")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 左圖：漂移比例圓餅圖
        summary = self.drift_results['summary']
        drifted = summary['drifted_features']
        no_drift = summary['total_features'] - drifted
        
        colors = ['#ff6b6b', '#51cf66']
        explode = (0.1, 0)
        
        axes[0].pie([drifted, no_drift], 
                    labels=['有漂移', '無漂移'],
                    autopct='%1.1f%%',
                    colors=colors,
                    explode=explode,
                    startangle=90)
        axes[0].set_title('特徵漂移比例', fontsize=14, fontweight='bold')
        
        # 右圖：特徵漂移狀態條形圖
        feature_names = []
        has_drift_values = []
        
        for feature, result in self.drift_results['feature_results'].items():
            feature_names.append(feature)
            has_drift_values.append(1 if result['has_drift'] else 0)
        
        colors_bar = ['#ff6b6b' if x == 1 else '#51cf66' for x in has_drift_values]
        
        y_pos = np.arange(len(feature_names))
        axes[1].barh(y_pos, has_drift_values, color=colors_bar)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(feature_names)
        axes[1].set_xlabel('漂移狀態 (0=無, 1=有)')
        axes[1].set_title('各特徵漂移狀態', fontsize=14, fontweight='bold')
        axes[1].set_xlim([0, 1.2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已儲存至: {save_path}")
        
        plt.show()
    
    def plot_numerical_drift(self, feature_name: str, figsize=(15, 5), save_path=None):
        """
        繪製數值型特徵的漂移分析圖
        
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
        
        ref_data = self.detector.reference_data[feature_name].dropna()
        
        # 如果有當前資料，從 detector 中取得
        # 注意：這裡需要從外部傳入當前資料
        # 為了示範，我們使用參考資料的統計
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 直方圖比較
        axes[0].hist(ref_data, bins=30, alpha=0.7, label='參考資料', color='blue', density=True)
        axes[0].set_xlabel('值')
        axes[0].set_ylabel('密度')
        axes[0].set_title(f'{feature_name} - 分布比較', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 箱型圖比較
        ref_stats = result['statistics']['reference']
        curr_stats = result['statistics']['current']
        
        box_data = [
            [ref_stats['min'], ref_stats['median'], ref_stats['max']],
            [curr_stats['min'], curr_stats['median'], curr_stats['max']]
        ]
        
        bp = axes[1].boxplot([ref_data], labels=['參考資料'], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[1].set_ylabel('值')
        axes[1].set_title(f'{feature_name} - 箱型圖', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # 統計量比較
        metrics = ['平均值', '標準差', '中位數']
        ref_values = [ref_stats['mean'], ref_stats['std'], ref_stats['median']]
        curr_values = [curr_stats['mean'], curr_stats['std'], curr_stats['median']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[2].bar(x - width/2, ref_values, width, label='參考資料', color='blue', alpha=0.7)
        axes[2].bar(x + width/2, curr_values, width, label='當前資料', color='red', alpha=0.7)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(metrics)
        axes[2].set_ylabel('值')
        axes[2].set_title(f'{feature_name} - 統計量比較', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis='y')
        
        # 添加漂移狀態標註
        drift_status = "⚠️ 偵測到漂移" if result['has_drift'] else "✓ 無漂移"
        fig.suptitle(f'特徵: {feature_name} ({drift_status})', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已儲存至: {save_path}")
        
        plt.show()
    
    def plot_categorical_drift(self, feature_name: str, figsize=(12, 6), save_path=None):
        """
        繪製類別型特徵的漂移分析圖
        
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
        
        # 合併所有類別
        all_categories = sorted(set(ref_dist.keys()) | set(curr_dist.keys()))
        
        ref_values = [ref_dist.get(cat, 0) for cat in all_categories]
        curr_values = [curr_dist.get(cat, 0) for cat in all_categories]
        
        # 正規化為比例
        ref_total = sum(ref_values)
        curr_total = sum(curr_values)
        ref_props = [v / ref_total for v in ref_values]
        curr_props = [v / curr_total for v in curr_values]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 左圖：條形圖比較
        x = np.arange(len(all_categories))
        width = 0.35
        
        axes[0].bar(x - width/2, ref_props, width, label='參考資料', color='blue', alpha=0.7)
        axes[0].bar(x + width/2, curr_props, width, label='當前資料', color='red', alpha=0.7)
        axes[0].set_xlabel('類別')
        axes[0].set_ylabel('比例')
        axes[0].set_title(f'{feature_name} - 類別分布比較', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(all_categories, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 右圖：差異熱圖
        diff = [curr_props[i] - ref_props[i] for i in range(len(all_categories))]
        colors = ['red' if d > 0 else 'blue' for d in diff]
        
        axes[1].barh(all_categories, diff, color=colors, alpha=0.7)
        axes[1].set_xlabel('比例差異 (當前 - 參考)')
        axes[1].set_title(f'{feature_name} - 分布變化', fontweight='bold')
        axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # 添加漂移狀態標註
        drift_status = "⚠️ 偵測到漂移" if result['has_drift'] else "✓ 無漂移"
        fig.suptitle(f'特徵: {feature_name} ({drift_status})', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已儲存至: {save_path}")
        
        plt.show()
    
    def plot_psi_heatmap(self, figsize=(10, 8), save_path=None):
        """
        繪製所有特徵的 PSI 值熱圖
        """
        if not self.drift_results:
            print("尚未執行漂移偵測")
            return
        
        feature_names = []
        psi_values = []
        
        for feature, result in self.drift_results['feature_results'].items():
            feature_names.append(feature)
            psi_values.append(result['tests']['psi']['value'])
        
        # 創建資料框
        df = pd.DataFrame({
            'Feature': feature_names,
            'PSI': psi_values
        }).sort_values('PSI', ascending=False)
        
        # 繪製橫條圖
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['#ff6b6b' if psi > 0.2 else '#ffd93d' if psi > 0.1 else '#51cf66' 
                  for psi in df['PSI']]
        
        bars = ax.barh(df['Feature'], df['PSI'], color=colors)
        ax.set_xlabel('PSI 值', fontweight='bold')
        ax.set_title('各特徵 PSI 值', fontsize=14, fontweight='bold')
        ax.axvline(x=0.1, color='orange', linestyle='--', linewidth=1, label='輕微變化閾值 (0.1)')
        ax.axvline(x=0.2, color='red', linestyle='--', linewidth=1, label='顯著漂移閾值 (0.2)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加數值標籤
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}',
                   ha='left', va='center', fontsize=9)
        
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
        
        # 繪製摘要
        summary_path = f"{save_dir}/drift_summary.png" if save_dir else None
        self.plot_drift_summary(save_path=summary_path)
        
        # 繪製 PSI 熱圖
        psi_path = f"{save_dir}/psi_heatmap.png" if save_dir else None
        self.plot_psi_heatmap(save_path=psi_path)
        
        # 繪製各特徵詳細圖
        for feature, result in self.drift_results['feature_results'].items():
            feature_path = f"{save_dir}/{feature}_drift.png" if save_dir else None
            
            if result['feature_type'] == 'numerical':
                self.plot_numerical_drift(feature, save_path=feature_path)
            else:
                self.plot_categorical_drift(feature, save_path=feature_path)


def create_drift_comparison_plot(ref_data: pd.DataFrame, 
                                 curr_data: pd.DataFrame,
                                 feature_name: str,
                                 figsize=(12, 4)):
    """
    快速創建單一特徵的漂移比較圖
    
    Parameters:
    -----------
    ref_data : pd.DataFrame
        參考資料集
    curr_data : pd.DataFrame
        當前資料集
    feature_name : str
        要比較的特徵名稱
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ref_values = ref_data[feature_name].dropna()
    curr_values = curr_data[feature_name].dropna()
    
    # 直方圖
    axes[0].hist(ref_values, bins=30, alpha=0.6, label='參考資料', color='blue', density=True)
    axes[0].hist(curr_values, bins=30, alpha=0.6, label='當前資料', color='red', density=True)
    axes[0].set_xlabel('值')
    axes[0].set_ylabel('密度')
    axes[0].set_title(f'{feature_name} - 分布比較')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 箱型圖
    axes[1].boxplot([ref_values, curr_values], 
                    labels=['參考資料', '當前資料'],
                    patch_artist=True)
    axes[1].set_ylabel('值')
    axes[1].set_title(f'{feature_name} - 箱型圖比較')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("視覺化模組已就緒")
