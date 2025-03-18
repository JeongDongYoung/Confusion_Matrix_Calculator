import pandas as pd
from model_evaluation import evaluate_model, print_results

def main():
    print("="*50)
    print("📌 KT Responsible AI Cetner RAI Tech team 📌")
    print("="*50)

    
    csv_file = input("Name of CSV file: ")
    
    try:
        df = pd.read_csv(csv_file)
        
        category_columns = [col for col in df.columns if 'category' in col.lower()]
        
        if category_columns:
            category_col = category_columns[0]
            categories = df[category_col].dropna().unique()
            
            print("\n=== Available Categories ===")
            for i, cat in enumerate(categories):
                print(f"{i+1}. {cat}")
            print("* You can also type 'total' for all data")
            print("="*30)
        
        category_name = input("Category: ")
        
        print("\n=== Column List ===")
        for i, col in enumerate(df.columns):
            print(f"{i+1}. {col}")
        print("="*30)
        
    except Exception as e:
        print(f"Error: {e}")
        return
    
    prediction_column = input("Model predict column: ")
    truth_column = input("Ground truth column: ")
    
    results = evaluate_model(csv_file, category_name, prediction_column, truth_column)
    
    print_results(results)
        
if __name__ == "__main__":
    main()