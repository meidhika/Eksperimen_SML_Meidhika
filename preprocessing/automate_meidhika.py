import pandas as pd
import os
from sklearn.model_selection import train_test_split

def run_preprocessing():
    print("Memulai proses preprocessing otomatis...")
    
    # Membaca data raw (Asumsi skrip dijalankan dari root folder repository)
    raw_data_path = 'credit_scoring_raw.csv'
    
    # Fallback jika dijalankan langsung dari dalam folder preprocessing
    if not os.path.exists(raw_data_path):
        raw_data_path = '../credit_scoring_raw.csv'
        
    df = pd.read_csv(raw_data_path)
    
    # 1. Drop kolom ID
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        
    # 2. Rename target kolom
    df = df.rename(columns={'default.payment.next.month': 'default'})
    
    # 3. Split Fitur dan Target
    X = df.drop(columns=['default'])
    y = df['default']
    
    # 4. Train-Test Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Menggabungkan kembali untuk disimpan
    train_data = X_train.copy()
    train_data['default'] = y_train
    test_data = X_test.copy()
    test_data['default'] = y_test
    
    # 6. Membuat folder output dan menyimpan file
    output_folder = 'preprocessing/credit_scoring_preprocessing'
    os.makedirs(output_folder, exist_ok=True)
    
    train_data.to_csv(f'{output_folder}/train.csv', index=False)
    test_data.to_csv(f'{output_folder}/test.csv', index=False)
    
    print(f"Preprocessing selesai! Data disimpan di '{output_folder}'")

if __name__ == "__main__":
    run_preprocessing()