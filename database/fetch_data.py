# database/fetch_data.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from database.db_connection import get_db_connection

# Define output directory and file path
OUTPUT_DIR = "data/fetched"
OUTPUT_FILE = "customer_features_fetched.csv"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)


def fetch_customer_features():
    """
    Fetch all customer features from the database and save to CSV
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get database connection
    connection = get_db_connection()
    
    if connection is None:
        print("Failed to establish database connection. Exiting.")
        return
    
    try:
        # Fetch data from customer_features table
        query = "SELECT * FROM customer_features"
        df = pd.read_sql(query, connection)
        
        # Save to CSV
        df.to_csv(OUTPUT_PATH, index=False)
        
        print(f"✓ Data fetched successfully!")
        print(f"✓ Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"✓ Saved to: {OUTPUT_PATH}")
        print(f"\nColumns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return None
        
    finally:
        connection.close()


def fetch_customer_features_with_filters(limit=None, columns=None):
    """
    Fetch customer features with optional filtering
    
    Args:
        limit (int): Maximum number of rows to fetch
        columns (list): Specific columns to fetch (None = all columns)
    """
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    connection = get_db_connection()
    
    if connection is None:
        print("Failed to establish database connection. Exiting.")
        return
    
    try:
        # Build query
        if columns:
            cols = ", ".join(columns)
            query = f"SELECT {cols} FROM customer_features"
        else:
            query = "SELECT * FROM customer_features"
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql(query, connection)
        
        # Save to CSV
        output_file = f"customer_features_fetched_{df.shape[0]}_rows.csv"
        output_path = os.path.join(OUTPUT_DIR, output_file)
        df.to_csv(output_path, index=False)
        
        print(f"✓ Data fetched successfully!")
        print(f"✓ Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"✓ Saved to: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return None
        
    finally:
        connection.close()


if __name__ == "__main__":
    # Option 1: Fetch all data
    fetch_customer_features()
    
    # Option 2: Fetch with limits and specific columns (uncomment to use)
    # columns_to_fetch = ["customer_id", "AccountWeeks", "DataUsage", "MonthlyCharge"]
    # fetch_customer_features_with_filters(limit=100, columns=columns_to_fetch)