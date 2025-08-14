# scripts/data_preprocessing.py
import pandas as pd

def load_and_clean_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)

    # Drop irrelevant columns based on correlation analysis
    drop_cols = ['condition', 'yr_built', 'month', 'yr_renovated', 'year']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Handle categorical / boolean values
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes

    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)

    return df

if __name__ == "__main__":
    cleaned_df = load_and_clean_data("data/kc_house_data.csv")
    cleaned_df.to_csv("data/cleaned_house_data.csv", index=False)
    print("✅ Data cleaned and saved to data/cleaned_house_data.csv")
