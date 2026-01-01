import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
import os
import sys

class FraudDetectionModel:
    """
    負責執行信用卡詐欺偵測模型的訓練與評估流程。
    採用 Scikit-Learn 的 RandomForest 演算法。
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_data(self):
        """
        讀取 CSV 資料集並進行基本的檢查。
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"錯誤: 找不到檔案 '{self.file_path}'。請確認檔案是否在目錄下。")

        try:
            print(f"正在讀取資料: {self.file_path} ...")
            self.df = pd.read_csv(self.file_path)
            print(f"資料讀取成功。資料維度: {self.df.shape}")
            
            # 基本檢查：確認必要欄位存在
            required_columns = ['Time', 'Amount', 'Class']
            if not all(col in self.df.columns for col in required_columns):
                raise ValueError(f"資料集缺少必要欄位，預期包含: {required_columns}")

        except Exception as e:
            print(f"讀取資料時發生未預期的錯誤: {e}")
            sys.exit(1)

    def preprocess_data(self):
        """
        資料前處理：
        1. 特徵縮放 (Scaling): 針對 'Time' 與 'Amount'。
        2. 切分訓練集與測試集。
        """
        if self.df is None:
            raise ValueError("資料尚未讀取，請先執行 load_data()")

        print("正在進行資料前處理...")

        # 1. 特徵縮放
        # 使用 RobustScaler 而非 StandardScaler，因為詐欺金額可能包含離群值 (Outliers)
        rob_scaler = RobustScaler()
        
        # 避免 SettingWithCopyWarning，使用 .copy()
        self.df['scaled_amount'] = rob_scaler.fit_transform(self.df['Amount'].values.reshape(-1, 1))
        self.df['scaled_time'] = rob_scaler.fit_transform(self.df['Time'].values.reshape(-1, 1))

        # 移除原始未縮放欄位，保留 V1-V28 (已經是 PCA 結果) 以及新的 scaled 欄位
        self.df.drop(['Time', 'Amount'], axis=1, inplace=True)

        # 2. 定義特徵 X 與 目標 y
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']

        # 3. 切分資料集
        # stratify=y 至關重要，確保訓練集和測試集中詐欺的比例一致 (因為這是極度不平衡資料)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"資料切分完成。訓練集樣本數: {len(self.X_train)}, 測試集樣本數: {len(self.X_test)}")

    def train_model(self):
        """
        訓練模型：
        使用 RandomForestClassifier。
        關鍵設定: class_weight='balanced' 用於自動調整權重以處理不平衡資料。
        """
        print("開始訓練模型 (Random Forest)... 這可能需要幾分鐘...")
        
        # class_weight='balanced' 會根據類別頻率自動給予少數類別(詐欺)較高的權重
        self.model = RandomForestClassifier(
            n_estimators=100,      # 樹的數量
            max_depth=10,          # 限制樹深防止過擬合
            class_weight='balanced', # 關鍵：處理不平衡資料
            random_state=42,
            n_jobs=-1              # 使用所有 CPU 核心加速
        )
        
        self.model.fit(self.X_train, self.y_train)
        print("模型訓練完成。")

    def evaluate_model(self):
        """
        評估模型表現。
        重點關注 Precision (精確率), Recall (召回率) 和 AUPRC。
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先執行 train_model()")

        print("\n--- 模型評估報告 ---")
        y_pred = self.model.predict(self.X_test)
        
        # 1. 混淆矩陣
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"混淆矩陣 (Confusion Matrix):\n{cm}")
        print("(Row 0: 正常, Row 1: 詐欺)")

        # 2. 詳細報告
        # 對於詐欺偵測，Recall (抓到多少詐欺) 通常比 Precision 更重要，但也需權衡誤報率
        print("\n分類報告 (Classification Report):")
        print(classification_report(self.y_test, y_pred, target_names=['Normal', 'Fraud']))

        # 3. AUPRC (Area Under the Precision-Recall Curve)
        # 對於極度不平衡資料，AUPRC 比 ROC-AUC 更具參考價值
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        auprc = average_precision_score(self.y_test, y_prob)
        print(f"AUPRC 分數: {auprc:.4f}")

if __name__ == "__main__":
    # 設定資料路徑 (假設與程式碼在同一層目錄)
    DATASET_PATH = 'creditcard.csv'
    
    # 初始化並執行流程
    detector = FraudDetectionModel(DATASET_PATH)
    
    try:
        detector.load_data()
        detector.preprocess_data()
        detector.train_model()
        detector.evaluate_model()
    except Exception as e:
        print(f"\n程式執行中斷: {e}")