"""
資料漂移檢測系統 - 配置文件
包含所有可調整的參數和閾值
"""

class DriftDetectionConfig:
    """
    漂移檢測配置類別
    
    使用方式:
    --------
    from drift_config import DriftDetectionConfig
    
    # 使用預設配置
    config = DriftDetectionConfig()
    
    # 或自訂配置
    config = DriftDetectionConfig.get_strict_config()
    """
    
    # ==================== 統計檢定閾值 ====================
    
    # 顯著性水準 (α)
    # 用於 KS Test, Chi-Square Test, Mann-Whitney Test 等
    SIGNIFICANCE_LEVEL = 0.05
    
    # 嚴格模式 (更容易檢測到漂移)
    SIGNIFICANCE_LEVEL_STRICT = 0.01
    
    # 寬鬆模式 (較不容易檢測到漂移)
    SIGNIFICANCE_LEVEL_LOOSE = 0.1
    
    # ==================== PSI 閾值 ====================
    
    # PSI (Population Stability Index) 閾值
    PSI_THRESHOLD_INSIGNIFICANT = 0.1    # 無顯著變化
    PSI_THRESHOLD_MODERATE = 0.2         # 顯著漂移
    PSI_THRESHOLD_SEVERE = 0.25          # 嚴重漂移
    
    # 預設使用的 PSI 閾值
    PSI_THRESHOLD_DEFAULT = PSI_THRESHOLD_MODERATE
    
    # ==================== 其他距離指標閾值 ====================
    
    # Jensen-Shannon Divergence 閾值
    JSD_THRESHOLD_LOW = 0.05        # 幾乎相同
    JSD_THRESHOLD_MODERATE = 0.1    # 輕微差異
    JSD_THRESHOLD_HIGH = 0.2        # 顯著差異
    
    # Total Variation Distance 閾值
    TVD_THRESHOLD_LOW = 0.1         # 非常相似
    TVD_THRESHOLD_MODERATE = 0.2    # 輕微差異
    TVD_THRESHOLD_HIGH = 0.3        # 顯著差異
    
    # Hellinger Distance 閾值
    HELLINGER_THRESHOLD_LOW = 0.1       # 非常相似
    HELLINGER_THRESHOLD_MODERATE = 0.2  # 輕微差異
    HELLINGER_THRESHOLD_HIGH = 0.3      # 顯著差異
    
    # ==================== 樣本大小要求 ====================
    
    # 最小樣本數
    # 低於此數量時會發出警告
    MIN_SAMPLE_SIZE = 30
    
    # 建議樣本數
    RECOMMENDED_SAMPLE_SIZE = 100
    
    # 大樣本數
    LARGE_SAMPLE_SIZE = 1000
    
    # ==================== 缺失值相關 ====================
    
    # 缺失值比例差異閾值
    # 超過此差異時認為缺失值有顯著變化
    MISSING_RATE_DIFF_THRESHOLD = 0.05  # 5%
    
    # 最大可接受缺失率
    MAX_ACCEPTABLE_MISSING_RATE = 0.3  # 30%
    
    # ==================== 分箱設定 ====================
    
    # PSI 計算時的分箱數量
    PSI_BINS = 10
    
    # JSD 計算時的分箱數量
    JSD_BINS = 10
    
    # 最小分箱數
    MIN_BINS = 5
    
    # 最大分箱數
    MAX_BINS = 20
    
    # ==================== 平滑參數 ====================
    
    # 用於避免 log(0) 的小常數
    EPSILON = 1e-4
    
    # ==================== 視覺化設定 ====================
    
    # 圖表大小
    FIGURE_SIZE_SMALL = (10, 6)
    FIGURE_SIZE_MEDIUM = (14, 8)
    FIGURE_SIZE_LARGE = (16, 10)
    
    # DPI 設定
    DPI_SCREEN = 100
    DPI_PRINT = 300
    
    # 顏色配置
    COLOR_NO_DRIFT = '#51cf66'      # 綠色 - 無漂移
    COLOR_MILD_DRIFT = '#ffd93d'    # 黃色 - 輕微漂移
    COLOR_DRIFT = '#ff6b6b'         # 紅色 - 有漂移
    
    # ==================== 報告設定 ====================
    
    # 報告格式
    REPORT_FORMAT_TEXT = 'text'
    REPORT_FORMAT_MARKDOWN = 'markdown'
    REPORT_FORMAT_HTML = 'html'
    
    # 預設報告格式
    DEFAULT_REPORT_FORMAT = REPORT_FORMAT_TEXT
    
    # ==================== 效能設定 ====================
    
    # 大數據集閾值
    # 超過此數量時可能需要抽樣
    LARGE_DATASET_THRESHOLD = 100000
    
    # 抽樣比例
    SAMPLING_RATIO = 0.1  # 10%
    
    # 最小抽樣數
    MIN_SAMPLING_SIZE = 10000
    
    # ==================== 預設配置方法 ====================
    
    @classmethod
    def get_default_config(cls) -> dict:
        """
        取得預設配置
        
        Returns:
        --------
        dict : 預設配置字典
        """
        return {
            'significance_level': cls.SIGNIFICANCE_LEVEL,
            'psi_threshold': cls.PSI_THRESHOLD_DEFAULT,
            'min_sample_size': cls.MIN_SAMPLE_SIZE,
            'jsd_threshold': cls.JSD_THRESHOLD_MODERATE,
            'tvd_threshold': cls.TVD_THRESHOLD_MODERATE,
            'hellinger_threshold': cls.HELLINGER_THRESHOLD_MODERATE,
            'missing_rate_diff_threshold': cls.MISSING_RATE_DIFF_THRESHOLD,
            'psi_bins': cls.PSI_BINS,
            'epsilon': cls.EPSILON
        }
    
    @classmethod
    def get_strict_config(cls) -> dict:
        """
        取得嚴格配置 (更容易檢測到漂移)
        
        Returns:
        --------
        dict : 嚴格配置字典
        """
        return {
            'significance_level': cls.SIGNIFICANCE_LEVEL_STRICT,
            'psi_threshold': cls.PSI_THRESHOLD_INSIGNIFICANT,
            'min_sample_size': cls.RECOMMENDED_SAMPLE_SIZE,
            'jsd_threshold': cls.JSD_THRESHOLD_LOW,
            'tvd_threshold': cls.TVD_THRESHOLD_LOW,
            'hellinger_threshold': cls.HELLINGER_THRESHOLD_LOW,
            'missing_rate_diff_threshold': 0.03,
            'psi_bins': cls.PSI_BINS,
            'epsilon': cls.EPSILON
        }
    
    @classmethod
    def get_loose_config(cls) -> dict:
        """
        取得寬鬆配置 (較不容易檢測到漂移)
        
        Returns:
        --------
        dict : 寬鬆配置字典
        """
        return {
            'significance_level': cls.SIGNIFICANCE_LEVEL_LOOSE,
            'psi_threshold': cls.PSI_THRESHOLD_SEVERE,
            'min_sample_size': cls.MIN_SAMPLE_SIZE,
            'jsd_threshold': cls.JSD_THRESHOLD_HIGH,
            'tvd_threshold': cls.TVD_THRESHOLD_HIGH,
            'hellinger_threshold': cls.HELLINGER_THRESHOLD_HIGH,
            'missing_rate_diff_threshold': 0.1,
            'psi_bins': cls.PSI_BINS,
            'epsilon': cls.EPSILON
        }
    
    @classmethod
    def get_production_config(cls) -> dict:
        """
        取得生產環境推薦配置
        
        Returns:
        --------
        dict : 生產環境配置字典
        """
        return {
            'significance_level': 0.01,  # 較嚴格
            'psi_threshold': 0.15,       # 中等
            'min_sample_size': cls.RECOMMENDED_SAMPLE_SIZE,
            'jsd_threshold': 0.08,
            'tvd_threshold': 0.15,
            'hellinger_threshold': 0.15,
            'missing_rate_diff_threshold': 0.05,
            'psi_bins': cls.PSI_BINS,
            'epsilon': cls.EPSILON
        }
    
    @classmethod
    def print_config(cls, config: dict = None):
        """
        列印配置資訊
        
        Parameters:
        -----------
        config : dict, optional
            要列印的配置,預設為 None (使用預設配置)
        """
        if config is None:
            config = cls.get_default_config()
        
        print("=" * 60)
        print("資料漂移檢測配置")
        print("=" * 60)
        for key, value in config.items():
            print(f"{key:30s}: {value}")
        print("=" * 60)


# ==================== 快速配置選擇器 ====================

class ConfigSelector:
    """
    配置選擇器 - 根據使用場景快速選擇配置
    """
    
    @staticmethod
    def select_by_scenario(scenario: str) -> dict:
        """
        根據使用場景選擇配置
        
        Parameters:
        -----------
        scenario : str
            使用場景,可選值:
            - 'development': 開發測試環境
            - 'production': 生產環境
            - 'monitoring': 監控系統
            - 'research': 研究分析
            - 'strict': 嚴格模式
            - 'loose': 寬鬆模式
        
        Returns:
        --------
        dict : 對應場景的配置
        """
        scenarios = {
            'development': DriftDetectionConfig.get_default_config(),
            'production': DriftDetectionConfig.get_production_config(),
            'monitoring': DriftDetectionConfig.get_strict_config(),
            'research': DriftDetectionConfig.get_default_config(),
            'strict': DriftDetectionConfig.get_strict_config(),
            'loose': DriftDetectionConfig.get_loose_config()
        }
        
        if scenario not in scenarios:
            print(f"⚠️  未知場景 '{scenario}',使用預設配置")
            return DriftDetectionConfig.get_default_config()
        
        config = scenarios[scenario]
        print(f"✅ 已選擇 '{scenario}' 場景配置")
        return config
    
    @staticmethod
    def select_by_data_size(n_samples: int) -> dict:
        """
        根據資料大小選擇配置
        
        Parameters:
        -----------
        n_samples : int
            樣本數量
        
        Returns:
        --------
        dict : 適合該資料大小的配置
        """
        config = DriftDetectionConfig.get_default_config()
        
        if n_samples < DriftDetectionConfig.MIN_SAMPLE_SIZE:
            print(f"⚠️  樣本數過小 ({n_samples}),結果可能不可靠")
            config['min_sample_size'] = n_samples // 2
        elif n_samples < DriftDetectionConfig.RECOMMENDED_SAMPLE_SIZE:
            print(f"⚠️  樣本數偏小 ({n_samples}),建議增加樣本")
        elif n_samples > DriftDetectionConfig.LARGE_DATASET_THRESHOLD:
            print(f"📊 大數據集 ({n_samples}),可考慮抽樣")
        else:
            print(f"✅ 樣本數適中 ({n_samples})")
        
        return config


# ==================== 使用範例 ====================

if __name__ == "__main__":
    print("🔧 資料漂移檢測配置系統\n")
    
    # 1. 顯示預設配置
    print("1. 預設配置:")
    DriftDetectionConfig.print_config()
    
    # 2. 顯示嚴格配置
    print("\n2. 嚴格配置:")
    DriftDetectionConfig.print_config(DriftDetectionConfig.get_strict_config())
    
    # 3. 使用場景選擇器
    print("\n3. 場景配置選擇:")
    config = ConfigSelector.select_by_scenario('production')
    DriftDetectionConfig.print_config(config)
    
    # 4. 根據資料大小選擇
    print("\n4. 資料大小配置選擇:")
    config = ConfigSelector.select_by_data_size(5000)
