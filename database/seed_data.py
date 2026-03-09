# database/seed_data.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from database.db_connection import get_db_connection
from config.db_config import DATASET_PATH, BATCH_SIZE


def seed_customer_data():

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset loaded: {df.shape}")

    connection = get_db_connection()
    
    if connection is None:
        print("Failed to establish database connection. Exiting.")
        return

    cursor = connection.cursor()

    insert_query = """
    INSERT INTO customer_features (
        AccountWeeks,
        ContractRenewal,
        DataPlan,
        DataUsage,
        CustServCalls,
        DayMins,
        DayCalls,
        MonthlyCharge,
        OverageFee,
        RoamMins
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    data = df.values.tolist()

    for i in range(0, len(data), BATCH_SIZE):

        batch = data[i:i+BATCH_SIZE]

        try:
            cursor.executemany(insert_query, batch)
            connection.commit()
            print(f"Inserted {i + len(batch)} rows")
        except Exception as e:
            connection.rollback()
            print(f"Error inserting batch: {e}")
            return

    cursor.close()
    connection.close()

    print("Data seeding completed")


if __name__ == "__main__":
    seed_customer_data()