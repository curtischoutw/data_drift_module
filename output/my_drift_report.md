# 資料漂移偵測報告

## 摘要
- **總特徵數**: 14
- **發生漂移的特徵數**: 9
- **漂移比例**: 64.29%

## 詳細結果
### num_late_payments - ⚠️ 有漂移
**類型**: numerical

#### 統計檢定結果
- **KS 檢定**: p-value = 0.0000 (Drift detected)
- **PSI**: 0.4772 (顯著漂移 (Significant drift))

#### 統計量變化
- **平均值**: 1.0215 → 1.9920
- **標準差**: 0.9927 → 1.4353

### data_date - ⚠️ 有漂移
**類型**: categorical

#### 統計檢定結果
- **Chi-Square 檢定**: p-value = 1.0000 (No drift)
- **PSI**: 3.3599 (顯著漂移 (Significant drift))

### loan_amount - ⚠️ 有漂移
**類型**: numerical

#### 統計檢定結果
- **KS 檢定**: p-value = 0.0000 (Drift detected)
- **PSI**: 0.3230 (顯著漂移 (Significant drift))

#### 統計量變化
- **平均值**: 149840.4982 → 181661.7511
- **標準差**: 51476.8798 → 56741.5957

### employment_type - ✓ 無漂移
**類型**: categorical

#### 統計檢定結果
- **Chi-Square 檢定**: p-value = 1.0000 (No drift)
- **PSI**: 0.0118 (無顯著變化 (No significant change))

### marital_status - ⚠️ 有漂移
**類型**: categorical

#### 統計檢定結果
- **Chi-Square 檢定**: p-value = 0.0000 (Drift detected)
- **PSI**: 0.1430 (輕微變化 (Slight change))

### employment_length_years - ⚠️ 有漂移
**類型**: numerical

#### 統計檢定結果
- **KS 檢定**: p-value = 0.0000 (Drift detected)
- **PSI**: 0.1769 (輕微變化 (Slight change))

#### 統計量變化
- **平均值**: 5.0305 → 5.9570
- **標準差**: 2.2045 → 2.3641

### num_credit_accounts - ✓ 無漂移
**類型**: numerical

#### 統計檢定結果
- **KS 檢定**: p-value = 0.6476 (No drift)
- **PSI**: 0.0134 (無顯著變化 (No significant change))

#### 統計量變化
- **平均值**: 7.9495 → 7.9790
- **標準差**: 2.8365 → 2.8957

### credit_score - ⚠️ 有漂移
**類型**: numerical

#### 統計檢定結果
- **KS 檢定**: p-value = 0.0000 (Drift detected)
- **PSI**: 0.1271 (輕微變化 (Slight change))

#### 統計量變化
- **平均值**: 679.6565 → 675.9040
- **標準差**: 60.6199 → 81.0329

### loan_purpose - ✓ 無漂移
**類型**: categorical

#### 統計檢定結果
- **Chi-Square 檢定**: p-value = 1.0000 (No drift)
- **PSI**: 0.0685 (無顯著變化 (No significant change))

### annual_income - ⚠️ 有漂移
**類型**: numerical

#### 統計檢定結果
- **KS 檢定**: p-value = 0.0000 (Drift detected)
- **PSI**: 0.3386 (顯著漂移 (Significant drift))

#### 統計量變化
- **平均值**: 62690.3479 → 75875.3312
- **標準差**: 21440.4408 → 24605.8539

### age - ⚠️ 有漂移
**類型**: numerical

#### 統計檢定結果
- **KS 檢定**: p-value = 0.0000 (Drift detected)
- **PSI**: 0.2691 (顯著漂移 (Significant drift))

#### 統計量變化
- **平均值**: 35.1070 → 30.8300
- **標準差**: 9.5259 → 10.4714

### housing_status - ✓ 無漂移
**類型**: categorical

#### 統計檢定結果
- **Chi-Square 檢定**: p-value = 1.0000 (No drift)
- **PSI**: 0.0066 (無顯著變化 (No significant change))

### debt_to_income_ratio - ⚠️ 有漂移
**類型**: numerical

#### 統計檢定結果
- **KS 檢定**: p-value = 0.0405 (Drift detected)
- **PSI**: 0.0223 (無顯著變化 (No significant change))

#### 統計量變化
- **平均值**: 0.2810 → 0.2923
- **標準差**: 0.1546 → 0.1640

### education_level - ✓ 無漂移
**類型**: categorical

#### 統計檢定結果
- **Chi-Square 檢定**: p-value = 1.0000 (No drift)
- **PSI**: 0.1815 (輕微變化 (Slight change))
