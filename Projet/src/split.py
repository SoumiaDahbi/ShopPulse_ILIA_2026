import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data():
    # 1. le chemin de dossier data 
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    original_path = os.path.join(data_dir, 'train.csv')
    df = pd.read_csv(original_path)
    print(f"Original data: {df.shape}")
    
    # split 1: Test Interne (20%) - pour estimation de performance
  
    X = df.drop(columns=['Purchased', 'SessionId'])
    Y = df['Purchased']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    
    # le fichier test_interne.csv 
    df_test = X_test.copy()
    df_test['Purchased'] = y_test
    df_test.to_csv(os.path.join(data_dir, 'test_interne.csv'), index=False)
    
    # split 2: Train (60%) / Validation (20% = 0.25 * 80% )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=0.25, random_state=42, stratify=y_temp 
    )
    
    # le fichier train_final.csv (60%)
    df_train = X_train.copy()
    df_train['Purchased'] = y_train
    df_train.to_csv(os.path.join(data_dir, 'train_final.csv'), index=False)
    
    # le fichier validation.csv (20% = 0.25 * 80% )
    df_val = X_val.copy()
    df_val['Purchased'] = y_val
    df_val.to_csv(os.path.join(data_dir, 'validation.csv'), index=False)
    

    # Vérifications

    print(f"\n Fichiers enregistrés f dossier 'data/':")
    print(f" - train_final.csv (60%): {len(df_train)} rows | Target: {df_train['Purchased'].mean():.4f}")
    print(f" - validation.csv (20%):  {len(df_val)} rows | Target: {df_val['Purchased'].mean():.4f}")
    print(f" - test_interne.csv (20%): {len(df_test)} rows | Target: {df_test['Purchased'].mean():.4f}")
    
if __name__ == '__main__':
    split_data()