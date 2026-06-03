import os
import pandas as pd
from sklearn.model_selection import train_test_split

def run_preprocessing():
    print("Memulai proses Data Preprocessing otomatis...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    raw_path = os.path.join(base_dir, 'BankNote_Authentication.csv')
    out_dir = os.path.join(current_dir, 'banknote_preprocessing')
    
    df = pd.read_csv(raw_path)
    
    X = df.drop(columns=['class'])
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_data = X_train.copy()
    train_data['class'] = y_train
    test_data = X_test.copy()
    test_data['class'] = y_test
    
    os.makedirs(out_dir, exist_ok=True)
    train_data.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    
    print(f"Selesai! Data tersimpan di folder: {out_dir}")

if __name__ == "__main__":
    run_preprocessing()