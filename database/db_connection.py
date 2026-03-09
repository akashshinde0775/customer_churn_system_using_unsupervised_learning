# database/db_connection.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pymysql
from config.db_config import DB_CONFIG
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_db_connection():
    """
    Get database connection using PyMySQL
    PyMySQL supports caching_sha2_password authentication
    """
    try:
        connection = pymysql.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"],
            port=DB_CONFIG["port"]
        )

        logger.info("✓ Database connection successful (PyMySQL)")
        return connection

    except pymysql.Error as err:
        logger.error(f"Database connection failed: {err}")
        return None


def close_connection(connection):
    """Close database connection"""
    if connection:
        connection.close()
        logger.info("✓ Database connection closed")