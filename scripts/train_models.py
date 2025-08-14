# scripts/train_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

def train_models():
    df = pd.read_csv("data/cleaned_house_data.csv")

    # Features and target
    X = df.drop(columns=['price'])
    y = df['price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_score = r2_score(y_test, lr_preds)

    # Decision Tree
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    dt_score = r2_score(y_test, dt_preds)

    print(f"Linear Regression R² Score: {lr_score:.4f}")
    print(f"Decision Tree R² Score: {dt_score:.4f}")

    return lr_score, dt_score

if __name__ == "__main__":
    train_models()
