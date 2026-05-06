import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils.preprocessing import load_and_clean, engineer_features
from models.model_selector import run_full_training

DATA_PATH   = os.path.join(os.path.dirname(__file__), 'data', 'sales_data.csv')
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), 'trained_models')

if __name__ == '__main__':
    print("=" * 60)
    print("  Sales Forecasting System — Training Pipeline")
    print("=" * 60)

    # 1. Load & clean
    df = load_and_clean(DATA_PATH)

    # 2. Feature engineering
    df = engineer_features(df)
    print(f"[Features] Columns: {list(df.columns)}")

    # 3. Train all models, select best per state
    results = run_full_training(df, output_dir=OUTPUT_DIR)

    print(f"\n Done! {len(results)} states trained.")
    print(f"   Results: {OUTPUT_DIR}/forecast_summary.json")
    print(f"   Start API: python api/app.py")
