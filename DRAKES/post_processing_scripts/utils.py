import os
import pandas as pd

def process_seq_data_directory(dir_name, target_col, process_fn):
    for fn in os.listdir(dir_name):
        file_path = os.path.join(dir_name, fn)
        if fn.lower().endswith(".csv"):
            print(f"Processing {fn}...")
            try:
                df = pd.read_csv(file_path, nrows=0)  # Read only the header
                if target_col in df.columns: # Don't re-compute if already done
                    print(f"{fn} already processed - Skipping...")
                else:
                    df = pd.read_csv(file_path)
                    value = process_fn(df)
                    df[target_col] = value
                    df.to_csv(file_path, index=False)
            except Exception as e: 
                print(e)
                pass
