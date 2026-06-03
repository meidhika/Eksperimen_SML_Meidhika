import os
import pandas as pd
from sklearn.model_selection import train_test_split

def run_preprocessing():
    print("Memulai proses Data Preprocessing otomatis...")
    
    # 1. Menentukan jalur folder
    # current_dir adalah folder 'preprocessing' tempat skrip ini berada
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # base_dir adalah root folder proyek
    base_dir = os.path.dirname(current_dir)
    
    # File mentah berada di root project
    raw_path = os.path.join(base_dir, 'BankNote_Authentication.csv')
    
    # Folder output yang baru: preprocessing/banknote_authentication
    out_dir = os.path.join(current_dir, 'banknote_authentication')
    
    # 2. Membaca dataset
    df = pd.read_csv(raw_path)
    
    # 3. Memisahkan Fitur dan Target
    X = df.drop(columns=['class'])
    y = df['class']
    
    # 4. Train-Test Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Menggabungkan kembali sebelum disimpan
    train_data = X_train.copy()
    train_data['class'] = y_train
    test_data = X_test.copy()
    test_data['class'] = y_test
    
    # 6. Membuat folder banknote_authentication jika belum ada, lalu menyimpan data
    os.makedirs(out_dir, exist_ok=True)
    train_data.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    
    print(f"Selesai! Data tersimpan di folder: {out_dir}")

if __name__ == "__main__":
    run_preprocessing()