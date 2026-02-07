import json
import pandas as pd
from datasets import Dataset

def prepare_data(input_file, output_file="data.json"):
    """
    Converts CSV to a Hugging Face compatible JSONL format for Mistral fine-tuning.
    """
    try:
        if input_file.endswith(".csv"):
            df = pd.read_csv(input_file)
        else:
            raise ValueError("Please provide a CSV file.")
            
        print(f"Columns found: {df.columns.tolist()}")
        
        # Check for expected columns
        if 'quote' not in df.columns or 'author' not in df.columns:
            print("Error: Expected 'quote' and 'author' columns.")
            return

        # Format: <s>[INST] {instruction} [/INST] {response} </s>
        # We will train it to generate quotes by specific authors.
        def format_row(row):
            instruction = f"Generate a quote by {row['author']}."
            if 'tags' in row and pd.notna(row['tags']):
                instruction = f"Generate a quote by {row['author']} about {row['tags']}."
                
            return f"<s>[INST] {instruction} [/INST] {row['quote']} </s>"

        df['text'] = df.apply(format_row, axis=1)
        
        # To JSONL
        df[['text']].to_json(output_file, orient='records', lines=True)
        print(f"Successfully converted {len(df)} rows to {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    import sys
    # Default to the known file path if not provided
    input_path = "quotes_cleaned.csv"
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        
    prepare_data(input_path)
