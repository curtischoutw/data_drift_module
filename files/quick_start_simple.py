"""
資料漂移檢測系統 - 快速入門腳本 (簡化版)
適合快速測試和基本使用
"""

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
import sys
proj_path = os.getenv('proj_path')
sys.path.append(proj_path)

import pandas as pd
from improved_data_drift_detector import ImprovedDataDriftDetector
from improved_drift_visualizer import ImprovedDriftVisualizer


# ========== 載入資料 ==========
print("=" * 70)
print("🚀 資料漂移檢測 - 快速開始")
print("=" * 70)
print()

# 載入您的資料（請修改路徑）
reference_data = pd.read_csv(f'{proj_path}/data/training_data.csv')
current_data = pd.read_csv(f'{proj_path}/data/production_data_with_drift.csv')

# 排除不需要的欄位（選擇性）
columns_to_exclude = ['id', 'timestamp', 'user_id']
reference_data = reference_data.drop(columns=columns_to_exclude, errors='ignore')
current_data = current_data.drop(columns=columns_to_exclude, errors='ignore')

# 確保欄位一致
common_columns = list(set(reference_data.columns) & set(current_data.columns))
reference_data = reference_data[common_columns]
current_data = current_data[common_columns]

print(f"✅ 資料載入完成")
print(f"   參考資料: {reference_data.shape}")
print(f"   當前資料: {current_data.shape}")
print(f"   分析特徵數: {len(common_columns)}")
print()


# ========== 執行檢測 ==========
print("🔍 執行漂移檢測...")

detector = ImprovedDataDriftDetector(
    reference_data=reference_data,
    threshold=0.05,
    psi_threshold=0.2
)

results = detector.detect_drift(current_data)
print("✅ 檢測完成！")
print()


# ========== 顯示結果 ==========
print("=" * 70)
print("📊 檢測結果")
print("=" * 70)

summary = results['summary']
print(f"總特徵數: {summary['total_features']}")
print(f"漂移特徵數: {summary['drifted_features']}")
print(f"漂移比例: {summary['drift_percentage']:.2f}%")
print()

# 顯示漂移特徵
drifted_features = detector.get_drifted_features()
if drifted_features:
    print(f"⚠️  發生漂移的特徵:")
    for feature in drifted_features:
        result = results['feature_results'][feature]
        psi = result['tests']['psi']['value']
        print(f"   • {feature} (PSI: {psi:.4f})")
    print()
    
    # 顯示前 3 個特徵的詳細資訊
    print("前 3 個特徵的詳細資訊:")
    print("-" * 70)
    
    for feature in drifted_features[:3]:
        result = results['feature_results'][feature]
        print(f"\n【{feature}】")
        print(f"類型: {result['feature_type']}")
        
        if result['feature_type'] == 'numerical':
            ref = result['statistics']['reference']
            cur = result['statistics']['current']
            print(f"平均值: {ref['mean']:.2f} → {cur['mean']:.2f}")
            print(f"標準差: {ref['std']:.2f} → {cur['std']:.2f}")
            print(f"PSI: {result['tests']['psi']['value']:.4f}")
        else:
            print(f"類別數: {result['statistics']['reference_categories']} → "
                  f"{result['statistics']['current_categories']}")
            
            new_cats = result['statistics'].get('new_categories', [])
            if new_cats:
                print(f"新類別: {new_cats}")
    
    print()
else:
    print("✅ 所有特徵均無顯著漂移")
    print()


# ========== 生成報告 ==========
print("=" * 70)
print("📝 生成報告")
print("=" * 70)

# 儲存報告
with open('drift_report.txt', 'w', encoding='utf-8') as f:
    f.write(detector.generate_report(output_format='text'))
print("✅ 報告已儲存: drift_report.txt")

# 儲存 Markdown 報告
with open('drift_report.md', 'w', encoding='utf-8') as f:
    f.write(detector.generate_report(output_format='markdown'))
print("✅ Markdown 已儲存: drift_report.md")
print()


# ========== 視覺化（可選） ==========
create_visualizations = True  # 設為 False 可跳過視覺化

if create_visualizations:
    try:
        print("=" * 70)
        print("🎨 生成視覺化")
        print("=" * 70)
        
        visualizer = ImprovedDriftVisualizer(detector, current_data)
        
        # 創建目錄
        os.makedirs('drift_plots', exist_ok=True)
        
        # 生成摘要圖
        visualizer.plot_drift_summary(save_path='drift_plots/summary.png')
        print("✅ 摘要圖已儲存")
        
        # 生成指標比較圖
        visualizer.plot_metrics_comparison(save_path='drift_plots/metrics.png')
        print("✅ 指標比較圖已儲存")
        
        # 為前 2 個漂移特徵生成詳細圖
        for feature in drifted_features[:2]:
            result = results['feature_results'][feature]
            if result['feature_type'] == 'numerical':
                visualizer.plot_numerical_drift(feature, 
                    save_path=f'drift_plots/{feature}.png')
            else:
                visualizer.plot_categorical_drift(feature,
                    save_path=f'drift_plots/{feature}.png')
            print(f"✅ {feature} 詳細圖已儲存")
        
        print()
        
    except Exception as e:
        print(f"⚠️  視覺化失敗: {str(e)}")
        print("請確認已安裝: pip install matplotlib seaborn")
        print()


# ========== 決策建議 ==========
print("=" * 70)
print("💡 建議行動")
print("=" * 70)

drift_pct = summary['drift_percentage']

if drift_pct >= 30:
    print("🚨 嚴重漂移 - 建議立即檢查資料管道並考慮重新訓練模型")
elif drift_pct >= 15:
    print("⚠️  中度漂移 - 建議密切監控並準備更新模型")
elif drift_pct >= 5:
    print("ℹ️  輕微漂移 - 建議持續觀察並記錄趨勢")
else:
    print("✅ 資料穩定 - 無需特別行動")

print()
print("=" * 70)
print("✅ 完成！")
print("=" * 70)
print()
print("已生成的檔案:")
print("  📄 drift_report.txt")
print("  📄 drift_report.md")
if create_visualizations:
    print("  📊 drift_plots/ (視覺化圖表)")
print()
