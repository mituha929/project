import pandas as pd
import numpy as np

def generate_paysim_with_cash_in():
    # 1. 設定檔案名稱 (請確認檔名正確)
    file_path = 'acct_transaction.csv'
    
    print(f"正在讀取檔案：{file_path} ...")
    
    try:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='cp950')
        print(f"✅ 成功讀取！資料共有 {len(df)} 筆")
    except FileNotFoundError:
        print("❌ 找不到檔案，請確認檔名是否正確。")
        return

    # --- 2. 修改通路對應 (關鍵修改處) ---
    # 這裡我們將 '06' 指定為 CASH_IN
    channel_map = {
        '01': 'CASH_OUT',   # ATM -> 提現
        '02': 'DEBIT',      # 臨櫃 -> 借記/臨櫃提款
        '03': 'TRANSFER',   # 行銀 -> 轉帳
        '04': 'TRANSFER',   # 網銀 -> 轉帳
        '05': 'PAYMENT',    # 語音 -> 支付
        '06': 'CASH_IN',    # <---【新增】將 eATM 定義為 存款 (CASH_IN)
        '07': 'PAYMENT',    # 電子支付
        '99': 'PAYMENT',
        'UNK': 'PAYMENT'
    }

    print("正在處理通路與時間...")

    # 處理 channel_type (擷取代碼)
    df['channel_str'] = df['channel_type'].astype(str)
    df['channel_code'] = df['channel_str'].apply(lambda x: x.split(':')[0].split('：')[0].strip())
    
    # 映射 Action
    df['action'] = df['channel_code'].map(channel_map).fillna('PAYMENT')

    # 處理時間 (Time) -> 小時 (Hour)
    df['txn_time'] = df['txn_time'].astype(str)
    df['dt'] = pd.to_datetime(df['txn_time'], format='%H:%M:%S', errors='coerce')
    
    # 如果 datetime 解析失敗太多，改用數字處理
    if df['dt'].isna().sum() > len(df) * 0.5:
        df['hour'] = pd.to_numeric(df['txn_time'], errors='coerce').fillna(0).astype(int) % 24
    else:
        df['hour'] = df['dt'].dt.hour.fillna(0).astype(int)

    # 處理日期與 Step
    df['day'] = pd.to_numeric(df['txn_date'], errors='coerce').fillna(1).astype(int)
    df['step'] = (df['day'] - 1) * 24 + df['hour']
    
    # 處理金額
    df['txn_amt'] = pd.to_numeric(df['txn_amt'], errors='coerce').fillna(0)

    # --- 3. 統計聚合 ---
    print("正在計算統計數據 (包含 CASH_IN)...")
    
    agg_df = df.groupby(['step', 'action']).agg({
        'day': 'first',
        'hour': 'first',
        'txn_amt': ['count', 'sum', 'mean', 'std']
    }).reset_index()

    # 整理欄位
    agg_df.columns = ['step', 'action', 'day', 'hour', 'count', 'sum', 'avg', 'std']
    agg_df['month'] = 1
    agg_df['std'] = agg_df['std'].fillna(0)

    # --- 4. 輸出結果 ---
    cols = ['action', 'month', 'day', 'hour', 'count', 'sum', 'avg', 'std', 'step']
    
    if agg_df.empty:
        print("❌ 生成失敗：資料為空。")
        return

    final_df = agg_df[cols]
    
    # 檢查是否真的有 CASH_IN
    if 'CASH_IN' in final_df['action'].values:
        print("✨ 成功偵測到 CASH_IN 交易數據！")
    else:
        print("⚠️ 警告：原始資料中沒有 '06' 代碼的交易，所以 CASH_IN 數量為 0。")
        print("   (建議：您可以手動將 channel_map 中的 '01' 或 '03' 也改為 CASH_IN 來測試)")

    output_file = 'aggregatedTransactions.csv'
    final_df.to_csv(output_file, index=False)
    
    print("-" * 30)
    print(f"✅ 檔案已生成：{output_file}")
    print("預覽包含 CASH_IN 的資料：")
    # 特別篩選出 CASH_IN 給你看，如果沒有則顯示頭 5 筆
    cash_in_preview = final_df[final_df['action'] == 'CASH_IN']
    if not cash_in_preview.empty:
        print(cash_in_preview.head())
    else:
        print(final_df.head())

# 執行
generate_paysim_with_cash_in()
#先pull ，分兩個資料夾，在push上去