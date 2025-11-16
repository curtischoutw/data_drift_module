"""
資料漂移檢測系統 - 快速入門腳本 (改進版)
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
from improved_data_drift_detector import ImprovedDataDriftDetector
from improved_drift_visualizer import ImprovedDriftVisualizer
from drift_config import ConfigSelector




# ========== 第一步：載入您的資料 ==========
# 請修改以下路徑為您的實際資料路徑

print("=" * 70)
print("📁 步驟 1: 載入資料")
print("=" * 70)

# 檔案載入
reference_data = pd.read_csv(f'{proj_path}/data/training_data.csv')                     # 訓練資料或基準資料
drift_data = pd.read_csv(f'{proj_path}/data/production_data_with_drift.csv')            # 有漂移的生產資料
nondrift_data = pd.read_csv(f'{proj_path}/data/production_data_no_drift.csv')           # 無漂移的生產資料

print(f"✅ 資料載入完成")
print(f"   參考資料: {reference_data.shape}")
print(f"   偏移資料: {drift_data.shape}")
print(f"   無偏移資料: {nondrift_data.shape}")
print()


# ========== 第二步：資料預處理（選擇性） ==========

print("=" * 70)
print("🔧 步驟 2: 資料預處理")
print("=" * 70)

# 1. 移除不需要檢查的欄位
columns_to_exclude = ['id', 'timestamp', 'user_id']  # 修改為您要排除的欄位
reference_data = reference_data.drop(columns=columns_to_exclude, errors='ignore')
drift_data = drift_data.drop(columns=columns_to_exclude, errors='ignore')
nondrift_data = nondrift_data.drop(columns=columns_to_exclude, errors='ignore')

# 2. 選擇要分析的資料集（有漂移或無漂移）
# 請根據您的需求選擇其中一個
current_data = drift_data  # 使用有漂移的資料
# current_data = nondrift_data  # 或使用無漂移的資料

# 3. 確保兩個資料集有相同的欄位
common_columns = list(set(reference_data.columns) & set(current_data.columns))
reference_data = reference_data[common_columns]
current_data = current_data[common_columns]

print(f"✅ 預處理完成")
print(f"   排除欄位: {columns_to_exclude}")
print(f"   分析特徵數: {len(common_columns)}")
print(f"   特徵列表: {common_columns}")
print()


# ========== 第三步：選擇檢測配置 ==========

print("=" * 70)
print("⚙️  步驟 3: 選擇檢測配置")
print("=" * 70)

# 方式 1: 使用預設配置
config = {
    'significance_level': 0.05,    # 顯著性水準 (可選: 0.01嚴格 / 0.05標準 / 0.10寬鬆)
    'psi_threshold': 0.2,          # PSI 閾值
    'min_sample_size': 30          # 最小樣本數
}

# 方式 2: 使用場景配置 (取消註解以使用)
# config = ConfigSelector.select_by_scenario('production')  # 生產環境
# config = ConfigSelector.select_by_scenario('strict')      # 嚴格模式
# config = ConfigSelector.select_by_scenario('loose')       # 寬鬆模式

print(f"✅ 配置已選擇")
print(f"   顯著性水準: {config['significance_level']}")
print(f"   PSI 閾值: {config['psi_threshold']}")
print(f"   最小樣本數: {config['min_sample_size']}")
print()


# ========== 第四步：執行漂移偵測 ==========

print("=" * 70)
print("🔍 步驟 4: 執行漂移偵測")
print("=" * 70)

# 創建偵測器
detector = ImprovedDataDriftDetector(
    reference_data=reference_data,
    threshold=config['significance_level'],
    min_sample_size=config['min_sample_size'],
    psi_threshold=config['psi_threshold']
)

print("開始執行漂移偵測...")
results = detector.detect_drift(current_data)
print("✅ 漂移偵測完成！")
print()


# ========== 第五步：查看整體結果 ==========

print("=" * 70)
print("📊 步驟 5: 整體結果摘要")
print("=" * 70)

summary = results['summary']
print(f"總特徵數: {summary['total_features']}")
print(f"發生漂移的特徵數: {summary['drifted_features']}")
print(f"漂移比例: {summary['drift_percentage']:.2f}%")

# 顯示警告訊息（如果有）
if summary['warnings']:
    print(f"\n⚠️  警告訊息:")
    for warning in summary['warnings']:
        print(f"   {warning}")

print()

# 列出發生漂移的特徵
drifted_features = detector.get_drifted_features()
if drifted_features:
    print(f"⚠️  發生漂移的特徵 ({len(drifted_features)} 個):")
    for feature in drifted_features:
        result = results['feature_results'][feature]
        psi_value = result['tests']['psi']['value']
        drift_reasons = ', '.join(result.get('drift_reasons', []))
        print(f"   • {feature}")
        print(f"     - PSI: {psi_value:.4f}")
        print(f"     - 漂移原因: {drift_reasons}")
    print()
else:
    print("✅ 所有特徵均無顯著漂移")
    print()


# ========== 第六步：特徵詳細分析 ==========

print("=" * 70)
print("📈 步驟 6: 特徵詳細分析")
print("=" * 70)

# 分析前 3 個發生漂移的特徵
if drifted_features:
    print(f"分析前 {min(3, len(drifted_features))} 個漂移特徵:\n")
    
    for i, feature in enumerate(drifted_features[:3], 1):
        result = results['feature_results'][feature]
        print(f"【特徵 {i}: {feature}】")
        print(f"  類型: {result['feature_type']}")
        print(f"  漂移狀態: {'❌ 有漂移' if result['has_drift'] else '✅ 無漂移'}")
        print(f"  漂移原因: {', '.join(result.get('drift_reasons', []))}")
        
        # 缺失值分析
        missing = result.get('missing_analysis', {})
        if missing and missing.get('reference_missing_rate', 0) > 0:
            print(f"\n  📊 缺失值分析:")
            print(f"     參考資料缺失率: {missing['reference_missing_rate']:.2%}")
            print(f"     當前資料缺失率: {missing['current_missing_rate']:.2%}")
            if missing['has_drift']:
                print(f"     ⚠️  缺失率有顯著變化!")
        
        # 統計檢定結果
        print(f"\n  🔬 統計檢定結果:")
        if result['feature_type'] == 'numerical':
            # 數值型特徵
            ks = result['tests']['ks_test']
            mw = result['tests']['mann_whitney']
            psi = result['tests']['psi']
            jsd = result['tests']['jensen_shannon_divergence']
            wass = result['tests']['wasserstein_distance']
            
            print(f"     • KS Test: p-value = {ks['p_value']:.4f} ({ks['interpretation']})")
            print(f"     • Mann-Whitney: p-value = {mw['p_value']:.4f} ({mw['interpretation']})")
            print(f"     • PSI: {psi['value']:.4f} ({psi['interpretation']})")
            print(f"     • JSD: {jsd['value']:.4f} ({jsd['interpretation']})")
            print(f"     • Wasserstein: {wass['value']:.4f} ({wass['interpretation']})")
            
            # 統計量變化
            ref_stats = result['statistics']['reference']
            curr_stats = result['statistics']['current']
            
            print(f"\n  📏 統計量變化:")
            print(f"     平均值: {ref_stats['mean']:.4f} → {curr_stats['mean']:.4f} " + 
                  f"(變化: {((curr_stats['mean'] - ref_stats['mean']) / ref_stats['mean'] * 100):.2f}%)")
            print(f"     標準差: {ref_stats['std']:.4f} → {curr_stats['std']:.4f}")
            print(f"     中位數: {ref_stats['median']:.4f} → {curr_stats['median']:.4f}")
            print(f"     偏度: {ref_stats['skewness']:.4f} → {curr_stats['skewness']:.4f}")
            print(f"     峰度: {ref_stats['kurtosis']:.4f} → {curr_stats['kurtosis']:.4f}")
            
        else:
            # 類別型特徵
            chi2 = result['tests']['chi_square_test']
            psi = result['tests']['psi']
            tvd = result['tests']['total_variation_distance']
            hellinger = result['tests']['hellinger_distance']
            
            print(f"     • Chi-Square: p-value = {chi2['p_value']:.4f} ({chi2['interpretation']})")
            print(f"     • PSI: {psi['value']:.4f} ({psi['interpretation']})")
            print(f"     • TVD: {tvd['value']:.4f} ({tvd['interpretation']})")
            print(f"     • Hellinger: {hellinger['value']:.4f} ({hellinger['interpretation']})")
            
            # 類別變化
            stats = result['statistics']
            new_cats = stats.get('new_categories', [])
            disappeared_cats = stats.get('disappeared_categories', [])
            
            print(f"\n  📊 類別分析:")
            print(f"     參考資料類別數: {stats['reference_categories']}")
            print(f"     當前資料類別數: {stats['current_categories']}")
            
            if new_cats:
                print(f"     ✨ 新出現類別: {new_cats}")
            if disappeared_cats:
                print(f"     ❌ 消失類別: {disappeared_cats}")
        
        print()
        print("-" * 70)
        print()
else:
    print("✅ 無特徵發生漂移，跳過詳細分析")
    print()


# ========== 第七步：缺失值整體分析 ==========

print("=" * 70)
print("🔍 步驟 7: 缺失值整體分析")
print("=" * 70)

has_missing_drift = False
for feature, result in results['feature_results'].items():
    missing = result.get('missing_analysis', {})
    if missing and missing.get('has_drift', False):
        if not has_missing_drift:
            print("⚠️  以下特徵的缺失值比例有顯著變化:\n")
            has_missing_drift = True
        
        print(f"  • {feature}:")
        print(f"    參考: {missing['reference_missing_rate']:.2%} → " + 
              f"當前: {missing['current_missing_rate']:.2%} " +
              f"(差異: {missing['difference']:.2%})")

if not has_missing_drift:
    print("✅ 所有特徵的缺失值比例無顯著變化")

print()


# ========== 第八步：多指標一致性檢查 ==========

print("=" * 70)
print("🎯 步驟 8: 多指標一致性檢查")
print("=" * 70)

print("檢查數值型特徵的多指標一致性:\n")

numerical_features = [f for f, r in results['feature_results'].items() 
                     if r['feature_type'] == 'numerical']

if numerical_features:
    for feature in numerical_features:
        result = results['feature_results'][feature]
        tests = result['tests']
        
        # 計算有多少指標檢測到漂移
        ks_drift = tests['ks_test']['p_value'] < config['significance_level']
        mw_drift = tests['mann_whitney']['p_value'] < config['significance_level']
        psi_drift = tests['psi']['value'] > config['psi_threshold']
        jsd_drift = tests['jensen_shannon_divergence']['value'] > 0.1
        
        drift_indicators = [ks_drift, mw_drift, psi_drift, jsd_drift]
        drift_count = sum(drift_indicators)
        
        # 只顯示有漂移的特徵
        if drift_count > 0:
            print(f"  • {feature}:")
            print(f"    檢測到漂移的指標數: {drift_count}/4")
            
            indicators = []
            if ks_drift: indicators.append("KS")
            if mw_drift: indicators.append("Mann-Whitney")
            if psi_drift: indicators.append("PSI")
            if jsd_drift: indicators.append("JSD")
            
            print(f"    漂移指標: {', '.join(indicators)}")
            
            if drift_count >= 3:
                print(f"    ✅ 多數指標確認有漂移 (高信心)")
            elif drift_count == 2:
                print(f"    ⚠️  部分指標檢測到漂移 (中等信心)")
            else:
                print(f"    ℹ️  僅單一指標檢測到漂移 (低信心)")
            print()
    
    if not any(sum([
        results['feature_results'][f]['tests']['ks_test']['p_value'] < config['significance_level'],
        results['feature_results'][f]['tests']['mann_whitney']['p_value'] < config['significance_level'],
        results['feature_results'][f]['tests']['psi']['value'] > config['psi_threshold'],
        results['feature_results'][f]['tests']['jensen_shannon_divergence']['value'] > 0.1
    ]) > 0 for f in numerical_features):
        print("  ✅ 所有數值型特徵無顯著漂移")
        print()
else:
    print("  ℹ️  無數值型特徵")
    print()


# ========== 第九步：生成報告 ==========

print("=" * 70)
print("📝 步驟 9: 生成報告")
print("=" * 70)

# 生成文字報告
text_report = detector.generate_report(output_format='text')
with open('drift_report.txt', 'w', encoding='utf-8') as f:
    f.write(text_report)
print("✅ 文字報告已儲存: drift_report.txt")

# 生成 Markdown 報告
md_report = detector.generate_report(output_format='markdown')
with open('drift_report.md', 'w', encoding='utf-8') as f:
    f.write(md_report)
print("✅ Markdown 報告已儲存: drift_report.md")
print()


# ========== 第十步：視覺化（選擇性） ==========

print("=" * 70)
print("🎨 步驟 10: 生成視覺化圖表")
print("=" * 70)

try:
    import matplotlib
    import seaborn
    
    print("正在生成視覺化圖表...\n")
    visualizer = ImprovedDriftVisualizer(detector, current_data)
    
    # 創建輸出目錄
    os.makedirs('drift_visualizations', exist_ok=True)
    
    # 1. 繪製漂移摘要圖
    print("  1/5 生成漂移摘要圖...")
    visualizer.plot_drift_summary(save_path='drift_visualizations/drift_summary.png')
    
    # 2. 繪製指標比較圖
    print("  2/5 生成指標比較圖...")
    visualizer.plot_metrics_comparison(save_path='drift_visualizations/metrics_comparison.png')
    
    # 3. 繪製前 2 個漂移特徵的詳細圖
    if drifted_features:
        for i, feature in enumerate(drifted_features[:2], 1):
            result = results['feature_results'][feature]
            print(f"  {2+i}/5 生成 {feature} 的詳細圖...")
            
            if result['feature_type'] == 'numerical':
                visualizer.plot_numerical_drift(
                    feature, 
                    save_path=f'drift_visualizations/{feature}_drift.png'
                )
            else:
                visualizer.plot_categorical_drift(
                    feature, 
                    save_path=f'drift_visualizations/{feature}_drift.png'
                )
    
    print(f"\n✅ 所有視覺化圖表已儲存至 drift_visualizations/ 目錄")
    
    # 可選: 生成所有特徵的詳細圖表（會花較長時間）
    # print("\n生成所有特徵的詳細圖表（這可能需要一些時間）...")
    # visualizer.plot_all_drifts(save_dir='drift_visualizations')
    
except ImportError:
    print("⚠️  視覺化功能需要安裝 matplotlib 和 seaborn")
    print("   執行: pip install matplotlib seaborn")
except Exception as e:
    print(f"⚠️  視覺化過程發生錯誤: {str(e)}")

print()


# ========== 第十一步：決策建議 ==========

print("=" * 70)
print("🎯 步驟 11: 決策建議")
print("=" * 70)

drift_percentage = results['summary']['drift_percentage']

if drift_percentage >= 30:
    print("🚨 嚴重漂移警告！")
    print()
    print("建議立即採取以下行動：")
    print("  1. 🔍 檢查資料收集管道")
    print("     - 驗證資料來源是否正常")
    print("     - 確認資料轉換流程無誤")
    print()
    print("  2. 📊 深入分析漂移原因")
    print("     - 區分資料品質問題 vs 真實分布變化")
    print("     - 檢視業務邏輯是否有變更")
    print()
    print("  3. ⚡ 評估模型效能")
    print("     - 使用最新資料測試模型準確度")
    print("     - 監控關鍵業務指標")
    print()
    print("  4. 🔄 考慮重新訓練模型")
    print("     - 使用包含新資料的訓練集")
    print("     - 評估是否需要調整特徵工程")
    
elif drift_percentage >= 15:
    print("⚠️  中度漂移警告")
    print()
    print("建議採取以下行動：")
    print("  1. 📈 密切監控模型效能")
    print("     - 增加監控頻率")
    print("     - 設置性能警報")
    print()
    print("  2. 🔎 分析主要漂移特徵")
    print("     - 重點關注影響最大的特徵")
    print("     - 了解業務背景變化")
    print()
    print("  3. 🛠️  準備重新訓練")
    print("     - 收集足夠的新資料")
    print("     - 規劃模型更新策略")
    
elif drift_percentage >= 5:
    print("ℹ️  輕微漂移")
    print()
    print("建議採取以下行動：")
    print("  1. 👁️  持續監控")
    print("     - 維持現有監控機制")
    print("     - 定期檢視漂移趨勢")
    print()
    print("  2. 📝 記錄漂移趨勢")
    print("     - 建立漂移歷史資料庫")
    print("     - 分析長期趨勢")
    print()
    print("  3. 🔔 設置警報閾值")
    print("     - 如果漂移持續增加，及時通知")
    
else:
    print("✅ 資料分布穩定")
    print()
    print("目前狀態：")
    print("  • 資料品質良好")
    print("  • 分布無顯著變化")
    print("  • 模型預期運作正常")
    print()
    print("建議：")
    print("  • 繼續正常監控週期")
    print("  • 無需特別行動")

print()


# ========== 第十二步：樣本檢查警告 ==========

print("=" * 70)
print("⚠️  步驟 12: 樣本檢查")
print("=" * 70)

has_sample_warning = False
for feature, result in results['feature_results'].items():
    sample_check = result.get('sample_check', {})
    if not sample_check.get('is_valid', True):
        if not has_sample_warning:
            print("以下特徵的樣本數可能不足，檢定結果僅供參考:\n")
            has_sample_warning = True
        
        print(f"  • {feature}:")
        for warning in sample_check.get('warnings', []):
            print(f"    - {warning}")
        print()

if not has_sample_warning:
    print("✅ 所有特徵的樣本數充足，檢定結果可靠")
    print()


# ========== 完成 ==========

print("=" * 70)
print("✅ 資料漂移檢測完成！")
print("=" * 70)
print()
print("生成的檔案:")
print("  📄 drift_report.txt - 文字格式報告")
print("  📄 drift_report.md - Markdown 格式報告")
print("  📊 drift_visualizations/ - 視覺化圖表目錄")
print()
print("下一步建議:")
print("  1. 查看詳細報告了解漂移情況")
print("  2. 檢視視覺化圖表進行直觀分析")
print("  3. 根據決策建議採取相應行動")
print()
print("=" * 70)
