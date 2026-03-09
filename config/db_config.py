# config/db_config.py

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",   
    "database": "customer_churn_system",
    "port": 3306
}

# Path of dataset to seed into database
DATASET_PATH = "data/raw/telecom_churn.csv"

# Batch insert size
BATCH_SIZE = 500