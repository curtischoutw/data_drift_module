"""
資料漂移檢測系統 - 快速入門腳本
請根據您的實際資料修改此腳本
"""

import os
import pandas as pd
from data_drift_module import DataDriftDetector, DriftVisualizer
from data_drift_module.config import ConfigSelector


# ========== 第一步：載入您的資料 ==========

print("=" * 70)
print("步驟 1: 載入資料")
print("=" * 70)

# 請修改以下路徑為您的實際資料路徑
reference_data = pd.read_csv('data/training_data.csv')
drift_data = pd.read_csv('data/production_data_with_drift.csv')
nondrift_data = pd.read_csv('data/production_data_no_drift.csv')

print(f"資料載入完成")
print(f"   參考資料: {reference_data.shape}")
print(f"   偏移資料: {drift_data.shape}")
print(f"   無偏移資料: {nondrift_data.shape}")
print()


# ========== 第二步：資料預處理（選擇性） ==========

print("=" * 70)
print("步驟 2: 資料預處理")
print("=" * 70)

columns_to_exclude = ['id', 'timestamp', 'user_id']  # 修改為您要排除的欄位
reference_data = reference_data.drop(columns=columns_to_exclude, errors='ignore')
drift_data = drift_data.drop(columns=columns_to_exclude, errors='ignore')
nondrift_data = nondrift_data.drop(columns=columns_to_exclude, errors='ignore')

# 選擇要分析的資料集（有漂移或無漂移）
current_data = drift_data  # 使用有漂移的資料
# current_data = nondrift_data  # 或使用無漂移的資料

# 確保兩個資料集有相同的欄位
common_columns = list(set(reference_data.columns) & set(current_data.columns))
reference_data = reference_data[common_columns]
current_data = current_data[common_columns]

print(f"預處理完成")
print(f"   排除欄位: {columns_to_exclude}")
print(f"   分析特徵數: {len(common_columns)}")
print(f"   特徵列表: {common_columns}")
print()


# ========== 第三步：選擇檢測配置 ==========

print("=" * 70)
print("步驟 3: 選擇檢測配置")
print("=" * 70)

# 方式 1: 直接指定
config = {
    'significance_level': 0.05,    # 顯著性水準 (可選: 0.01嚴格 / 0.05標準 / 0.10寬鬆)
    'psi_threshold': 0.2,          # PSI 閾值
    'min_sample_size': 30          # 最小樣本數
}

# 方式 2: 使用場景配置（取消註解以使用）
# config = ConfigSelector.select_by_scenario('production')  # 生產環境
# config = ConfigSelector.select_by_scenario('strict')      # 嚴格模式
# config = ConfigSelector.select_by_scenario('loose')       # 寬鬆模式

print(f"配置已選擇")
print(f"   顯著性水準: {config['significance_level']}")
print(f"   PSI 閾值: {config['psi_threshold']}")
print(f"   最小樣本數: {config['min_sample_size']}")
print()


# ========== 第四步：執行漂移偵測 ==========

print("=" * 70)
print("步驟 4: 執行漂移偵測")
print("=" * 70)

detector = DataDriftDetector(
    reference_data=reference_data,
    threshold=config['significance_level'],
    min_sample_size=config['min_sample_size'],
    psi_threshold=config['psi_threshold']
    # enable_mann_whitney=True,     # 啟用可選的 Mann-Whitney 檢定
    # enable_anderson_darling=True  # 啟用可選的 Anderson-Darling 檢定
)

print("開始執行漂移偵測...")
results = detector.detect_drift(current_data)
print("漂移偵測完成！")
print()


# ========== 第五步：查看整體結果 ==========

print("=" * 70)
print("步驟 5: 整體結果摘要")
print("=" * 70)

summary = results['summary']
print(f"總特徵數: {summary['total_features']}")
print(f"發生漂移的特徵數: {summary['drifted_features']}")
print(f"漂移比例: {summary['drift_percentage']:.2f}%")

if summary['warnings']:
    print(f"\n警告訊息:")
    for warning in summary['warnings']:
        print(f"   {warning}")

print()

drifted_features = detector.get_drifted_features()
if drifted_features:
    print(f"發生漂移的特徵 ({len(drifted_features)} 個):")
    for feature in drifted_features:
        result = results['feature_results'][feature]
        psi_value = result['tests']['psi']['value']
        drift_reasons = ', '.join(result.get('drift_reasons', []))
        print(f"   • {feature}")
        print(f"     - PSI: {psi_value:.4f}")
        print(f"     - 漂移原因: {drift_reasons}")
    print()
else:
    print("所有特徵均無顯著漂移")
    print()


# ========== 第六步：特徵詳細分析 ==========

print("=" * 70)
print("步驟 6: 特徵詳細分析（前 3 個漂移特徵）")
print("=" * 70)

if drifted_features:
    for i, feature in enumerate(drifted_features[:3], 1):
        result = results['feature_results'][feature]
        print(f"【特徵 {i}: {feature}】")
        print(f"  類型: {result['feature_type']}")
        print(f"  漂移狀態: {'有漂移' if result['has_drift'] else '無漂移'}")
        print(f"  漂移原因: {', '.join(result.get('drift_reasons', []))}")

        missing = result.get('missing_analysis', {})
        if missing and missing.get('reference_missing_rate', 0) > 0:
            print(f"\n  缺失值分析:")
            print(f"     參考資料缺失率: {missing['reference_missing_rate']:.2%}")
            print(f"     當前資料缺失率: {missing['current_missing_rate']:.2%}")
            if missing['has_drift']:
                print(f"     缺失率有顯著變化!")

        print(f"\n  統計檢定結果:")
        if result['feature_type'] == 'numerical':
            ks = result['tests']['ks_test']
            psi = result['tests']['psi']
            jsd = result['tests']['jensen_shannon_divergence']
            wass = result['tests']['wasserstein_distance']
            print(f"     • KS Test: p-value = {ks['p_value']:.4f} ({ks['interpretation']})")
            print(f"     • PSI: {psi['value']:.4f} ({psi['interpretation']})")
            print(f"     • JSD: {jsd['value']:.4f} ({jsd['interpretation']})")
            print(f"     • Wasserstein: {wass['value']:.4f} ({wass['interpretation']}) [輔助]")

            ref_stats = result['statistics']['reference']
            curr_stats = result['statistics']['current']
            print(f"\n  統計量變化:")
            print(f"     平均值: {ref_stats['mean']:.4f} → {curr_stats['mean']:.4f}")
            print(f"     標準差: {ref_stats['std']:.4f} → {curr_stats['std']:.4f}")
            print(f"     中位數: {ref_stats['median']:.4f} → {curr_stats['median']:.4f}")
        else:
            chi2 = result['tests']['chi_square_test']
            psi = result['tests']['psi']
            print(f"     • Chi-Square: p-value = {chi2['p_value']:.4f} ({chi2['interpretation']})")
            print(f"     • PSI: {psi['value']:.4f} ({psi['interpretation']})")

            cat_stats = result['statistics']
            new_cats = cat_stats.get('new_categories', [])
            disappeared_cats = cat_stats.get('disappeared_categories', [])
            print(f"\n  類別分析:")
            print(f"     參考資料類別數: {cat_stats['reference_categories']}")
            print(f"     當前資料類別數: {cat_stats['current_categories']}")
            if new_cats:
                print(f"     新出現類別: {new_cats}")
            if disappeared_cats:
                print(f"     消失類別: {disappeared_cats}")

        print()
        print("-" * 70)
        print()
else:
    print("無特徵發生漂移，跳過詳細分析")
    print()


# ========== 第七步：缺失值整體分析 ==========

print("=" * 70)
print("步驟 7: 缺失值整體分析")
print("=" * 70)

has_missing_drift = False
for feature, result in results['feature_results'].items():
    missing = result.get('missing_analysis', {})
    if missing and missing.get('has_drift', False):
        if not has_missing_drift:
            print("以下特徵的缺失值比例有顯著變化:\n")
            has_missing_drift = True
        print(f"  • {feature}:")
        print(f"    參考: {missing['reference_missing_rate']:.2%} → "
              f"當前: {missing['current_missing_rate']:.2%} "
              f"(差異: {missing['difference']:.2%})")

if not has_missing_drift:
    print("所有特徵的缺失值比例無顯著變化")

print()


# ========== 第八步：生成報告 ==========

print("=" * 70)
print("步驟 8: 生成報告")
print("=" * 70)

text_report = detector.generate_report(output_format='text')
with open('drift_report.txt', 'w', encoding='utf-8') as f:
    f.write(text_report)
print("文字報告已儲存: drift_report.txt")

md_report = detector.generate_report(output_format='markdown')
with open('drift_report.md', 'w', encoding='utf-8') as f:
    f.write(md_report)
print("Markdown 報告已儲存: drift_report.md")
print()


# ========== 第九步：視覺化（選擇性） ==========

print("=" * 70)
print("步驟 9: 生成視覺化圖表")
print("=" * 70)

try:
    print("正在生成視覺化圖表...\n")
    visualizer = DriftVisualizer(detector, current_data)

    os.makedirs('drift_visualizations', exist_ok=True)

    print("  1/3 生成漂移摘要圖...")
    visualizer.plot_drift_summary(save_path='drift_visualizations/drift_summary.png')

    print("  2/3 生成指標比較圖...")
    visualizer.plot_metrics_comparison(save_path='drift_visualizations/metrics_comparison.png')

    if drifted_features:
        for i, feature in enumerate(drifted_features[:2], 1):
            result = results['feature_results'][feature]
            print(f"  {2+i}/3+ 生成 {feature} 的詳細圖...")
            if result['feature_type'] == 'numerical':
                visualizer.plot_numerical_drift(
                    feature, save_path=f'drift_visualizations/{feature}_drift.png'
                )
            else:
                visualizer.plot_categorical_drift(
                    feature, save_path=f'drift_visualizations/{feature}_drift.png'
                )

    print(f"\n所有視覺化圖表已儲存至 drift_visualizations/ 目錄")

except ImportError:
    print("視覺化功能需要安裝 matplotlib 和 seaborn")
    print("執行: pip install matplotlib seaborn")
except Exception as e:
    print(f"視覺化過程發生錯誤: {str(e)}")

print()


# ========== 第十步：決策建議 ==========

print("=" * 70)
print("步驟 10: 決策建議")
print("=" * 70)

drift_percentage = results['summary']['drift_percentage']

if drift_percentage >= 30:
    print("嚴重漂移警告！建議立即採取以下行動：")
    print("  1. 檢查資料收集管道（驗證資料來源、確認轉換流程）")
    print("  2. 深入分析漂移原因（資料品質問題 vs 真實分布變化）")
    print("  3. 評估模型效能（使用最新資料測試準確度）")
    print("  4. 考慮重新訓練模型")
elif drift_percentage >= 15:
    print("中度漂移警告。建議採取以下行動：")
    print("  1. 密切監控模型效能（增加監控頻率、設置警報）")
    print("  2. 分析主要漂移特徵（了解業務背景變化）")
    print("  3. 準備重新訓練（收集足夠新資料、規劃更新策略）")
elif drift_percentage >= 5:
    print("輕微漂移。建議採取以下行動：")
    print("  1. 持續監控（維持現有機制、定期檢視趨勢）")
    print("  2. 記錄漂移趨勢（建立歷史資料庫、分析長期趨勢）")
    print("  3. 設置警報閾值（漂移持續增加時及時通知）")
else:
    print("資料分布穩定")
    print("  • 資料品質良好，分布無顯著變化")
    print("  • 繼續正常監控週期，無需特別行動")

print()


# ========== 完成 ==========

print("=" * 70)
print("資料漂移檢測完成！")
print("=" * 70)
print()
print("生成的檔案:")
print("  drift_report.txt - 文字格式報告")
print("  drift_report.md - Markdown 格式報告")
print("  drift_visualizations/ - 視覺化圖表目錄")
print()
