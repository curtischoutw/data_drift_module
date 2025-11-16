"""
快速入門腳本模板
請根據您的實際資料修改此腳本
"""

from dotenv import load_dotenv, find_dotenv
# 載入 .env 檔案
load_dotenv(find_dotenv())

import os
import sys
proj_path = os.getenv('proj_path')
sys.path.append(proj_path)

import pandas as pd
from dependencies import DataDriftDetector
from dependencies import DriftVisualizer




# ========== 第一步：載入您的資料 ==========
# 請修改以下路徑為您的實際資料路徑

# 檔案載入
reference_data = pd.read_csv(f'{proj_path}/data/training_data.csv')                     # 訓練資料或基準資料
current_data = pd.read_csv(f'{proj_path}/data/production_data_with_drift.csv')            # 當前生產資料

print(f"  資料載入完成")
print(f"  參考資料: {reference_data.shape}")
print(f"  偏移資料: {current_data.shape}")
print()


# ========== 第二步：資料預處理（選擇性） ==========

# 1. 移除不需要檢查的欄位
columns_to_exclude = ['id', 'timestamp', 'user_id']  # 修改為您要排除的欄位
reference_data = reference_data.drop(columns = columns_to_exclude, errors = 'ignore')
current_data = current_data.drop(columns = columns_to_exclude, errors = 'ignore')

# 2. 確保兩個資料集有相同的欄位
common_columns = list(set(reference_data.columns) & set(current_data.columns))
reference_data = reference_data[common_columns]
current_data = current_data[common_columns]

print(f"  預處理完成，共 {len(common_columns)} 個特徵將被檢查")
print()


# ========== 第三步：執行漂移偵測 ==========

# 創建偵測器
detector = DataDriftDetector(
    reference_data=reference_data,
    threshold=0.05  # 可調整: 0.01(嚴格) / 0.05(標準) / 0.10(寬鬆)
)

print("🔍 開始執行漂移偵測...")
results = detector.detect_drift(current_data)
print()


# ========== 第四步：查看結果 ==========

print("=" * 60)
print("📊 漂移偵測結果摘要")
print("=" * 60)
print(f"總特徵數: {results['summary']['total_features']}")
print(f"發生漂移的特徵數: {results['summary']['drifted_features']}")
print(f"漂移比例: {results['summary']['drift_percentage']:.2f}%")
print()

# 列出發生漂移的特徵
drifted_features = detector.get_drifted_features()
if drifted_features:
    print("⚠️  發生漂移的特徵:")
    for feature in drifted_features:
        result = results['feature_results'][feature]
        psi_value = result['tests']['psi']['value']
        print(f"  - {feature} (PSI: {psi_value:.4f})")
    print()
else:
    print("✓ 所有特徵均無顯著漂移")
    print()


# ========== 第五步：詳細分析（選擇性） ==========

# 分析前 3 個發生漂移的特徵
if drifted_features:
    print("=" * 60)
    print("📈 前 3 個漂移特徵的詳細分析")
    print("=" * 60)
    print()
    
    for feature in drifted_features[:3]:
        result = results['feature_results'][feature]
        print(f"特徵: {feature}")
        print(f"  類型: {result['feature_type']}")
        
        if result['feature_type'] == 'numerical':
            # 數值型特徵
            ks = result['tests']['ks_test']
            psi = result['tests']['psi']
            ref_stats = result['statistics']['reference']
            curr_stats = result['statistics']['current']
            
            print(f"  KS 檢定 p-value: {ks['p_value']:.4f}")
            print(f"  PSI 值: {psi['value']:.4f} - {psi['interpretation']}")
            print(f"  平均值: {ref_stats['mean']:.2f} → {curr_stats['mean']:.2f}")
            print(f"  標準差: {ref_stats['std']:.2f} → {curr_stats['std']:.2f}")
        else:
            # 類別型特徵
            chi2 = result['tests']['chi_square_test']
            psi = result['tests']['psi']
            
            print(f"  Chi-Square p-value: {chi2['p_value']:.4f}")
            print(f"  PSI 值: {psi['value']:.4f} - {psi['interpretation']}")
        
        print()


# ========== 第六步：生成報告 ==========

# 生成文字報告
text_report = detector.generate_report(output_format='text')
with open('my_drift_report.txt', 'w', encoding='utf-8') as f:
    f.write(text_report)
print("✓ 文字報告已儲存: my_drift_report.txt")

# 生成 Markdown 報告
md_report = detector.generate_report(output_format='markdown')
with open('my_drift_report.md', 'w', encoding='utf-8') as f:
    f.write(md_report)
print("✓ Markdown 報告已儲存: my_drift_report.md")
print()


# ========== 第七步：視覺化（選擇性） ==========

try:
    print("🎨 生成視覺化圖表...")
    visualizer = DriftVisualizer(detector)
    
    # 1. 繪製摘要圖
    visualizer.plot_drift_summary(save_path='drift_summary.png')
    
    # 2. 繪製 PSI 熱圖
    visualizer.plot_psi_heatmap(save_path='psi_heatmap.png')
    
    # 3. 繪製前 2 個漂移特徵的詳細圖
    for i, feature in enumerate(drifted_features[:2]):
        result = results['feature_results'][feature]
        if result['feature_type'] == 'numerical':
            visualizer.plot_numerical_drift(feature, save_path=f'{feature}_drift.png')
        else:
            visualizer.plot_categorical_drift(feature, save_path=f'{feature}_drift.png')
    
    print("✓ 視覺化圖表已生成")
    
except ImportError:
    print("⚠️  視覺化功能需要安裝 matplotlib 和 seaborn")
    print("   執行: pip install matplotlib seaborn")

print()


# ========== 第八步：決策邏輯（根據您的需求自訂） ==========

print("=" * 60)
print("🎯 建議行動")
print("=" * 60)

drift_percentage = results['summary']['drift_percentage']

if drift_percentage >= 30:
    print("🚨 嚴重漂移警告！")
    print("   建議立即採取以下行動：")
    print("   1. 檢查資料收集管道是否正常")
    print("   2. 分析漂移原因（資料品質 vs 真實分布變化）")
    print("   3. 評估模型效能是否下降")
    print("   4. 考慮重新訓練模型")
    
elif drift_percentage >= 15:
    print("⚠️  中度漂移警告")
    print("   建議採取以下行動：")
    print("   1. 密切監控模型效能")
    print("   2. 分析主要漂移特徵")
    print("   3. 準備重新訓練模型")
    
elif drift_percentage >= 5:
    print("ℹ️  輕微漂移")
    print("   建議採取以下行動：")
    print("   1. 持續監控")
    print("   2. 記錄漂移趨勢")
    
else:
    print("✓ 資料分布穩定")
    print("   無需特別行動，繼續正常監控")

print()
print("=" * 60)
print("✅ 漂移偵測完成！")
print("=" * 60)
