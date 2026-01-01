import pandas as pd
import numpy as np
import os
import sys

# Scikit-Learn 相關
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score

# 進階套件：XGBoost 與 Imbalanced-learn
try:
    import xgboost as xgb
    from imblearn.over_sampling import SMOTE
except ImportError as e:
    print("錯誤：缺少必要套件。請執行: pip install xgboost imbalanced-learn")
    sys.exit(1)

class AdvancedFraudDetection:
    """
    進階信用卡詐欺偵測模型。
    特點：
    1. 使用 XGBoost 提升分類效能。
    2. 使用 SMOTE 進行訓練集的過採樣 (Over-sampling) 以解決不平衡問題。
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        # 原始切分資料
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        # SMOTE 處理後的訓練資料
        self.X_train_resampled = None
        self.y_train_resampled = None
        
        self.model = None

    def load_and_preprocess(self):
        """
        讀取資料、檢查、縮放特徵並進行切分。
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"錯誤: 找不到檔案 '{self.file_path}'")

        print(f"1. 讀取資料集: {self.file_path} ...")
        self.df = pd.read_csv(self.file_path)
        
        # 簡單檢查
        if 'Class' not in self.df.columns:
            raise ValueError("資料集缺少 'Class' 欄位")

        # --- 特徵工程 ---
        print("2. 執行特徵縮放 (RobustScaler)...")
        rob_scaler = RobustScaler()
        self.df['scaled_amount'] = rob_scaler.fit_transform(self.df['Amount'].values.reshape(-1, 1))
        self.df['scaled_time'] = rob_scaler.fit_transform(self.df['Time'].values.reshape(-1, 1))
        self.df.drop(['Time', 'Amount'], axis=1, inplace=True)

        # --- 資料切分 ---
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']

        # 這裡必須先切分，確保測試集完全純淨
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"   資料切分完成。原始訓練集分佈: {np.bincount(self.y_train)}")

    def apply_smote(self):
        """
        關鍵步驟：僅對「訓練集」應用 SMOTE 生成合成樣本。
        """
        if self.X_train is None:
            raise ValueError("請先執行 load_and_preprocess")

        print("3. 正在應用 SMOTE 進行過採樣 (這可能需要一點時間)...")
        
        # sampling_strategy=0.1 表示將少數類別合成到多數類別數量的 10%
        # 你也可以設為 'auto' (1:1)，但在詐欺偵測中，1:10 (0.1) 通常效果就不錯且訓練較快
        smote = SMOTE(sampling_strategy=0.2, random_state=42)
        
        self.X_train_resampled, self.y_train_resampled = smote.fit_resample(self.X_train, self.y_train)
        
        print(f"   SMOTE 完成。")
        print(f"   處理前 詐欺筆數: {sum(self.y_train == 1)}")
        print(f"   處理後 詐欺筆數: {sum(self.y_train_resampled == 1)}")

    def train_xgboost(self):
        """
        使用 XGBoost 進行訓練。
        """
        print("4. 開始訓練 XGBoost 模型...")
        
        # 初始化 XGBoost 分類器
        # scale_pos_weight 在使用 SMOTE 後通常設為 1 即可，因為資料已經比較平衡
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,             # 樹的深度，過深容易過擬合
            objective='binary:logistic',
            eval_metric='aucpr',     # 使用 AUPRC 作為評估指標
            use_label_encoder=False,
            n_jobs=-1,
            random_state=42
        )

        self.model.fit(self.X_train_resampled, self.y_train_resampled)
        print("   模型訓練完成。")

    def evaluate(self):
        """
        輸出詳細評估報告。
        """
        print("\n--- 最終評估報告 (基於純淨測試集) ---")
        
        # 預測類別
        y_pred = self.model.predict(self.X_test)
        # 預測機率 (用於計算 AUPRC)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        # 混淆矩陣
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"混淆矩陣:\n{cm}")
        
        # 分類報告
        print("\n詳細數據:")
        print(classification_report(self.y_test, y_pred, target_names=['Normal', 'Fraud']))
        
        # AUPRC
        auprc = average_precision_score(self.y_test, y_prob)
        print(f"AUPRC 分數: {auprc:.4f} (越接近 1 越好)")

if __name__ == "__main__":
    DATASET_PATH = 'creditcard.csv'
    
    detector = AdvancedFraudDetection(DATASET_PATH)
    
    try:
        detector.load_and_preprocess() # 1. 讀取與切分
        detector.apply_smote()         # 2. SMOTE 生成樣本 (只對訓練集)
        detector.train_xgboost()       # 3. XGBoost 訓練
        detector.evaluate()            # 4. 評估
    except Exception as e:
        print(f"\n執行過程中發生錯誤: {e}")