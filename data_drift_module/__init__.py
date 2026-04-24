"""
資料漂移偵測系統 (Data Drift Detection System)
"""

from data_drift_module.detector import DataDriftDetector
from data_drift_module.visualizer import DriftVisualizer
from data_drift_module.config import DriftDetectionConfig, ConfigSelector

__version__ = "1.0.0"
__all__ = ["DataDriftDetector", "DriftVisualizer", "DriftDetectionConfig", "ConfigSelector"]
