from fastapi import APIRouter, HTTPException
from database.db_connection import get_db_connection, close_connection
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/risk-distribution")
async def get_risk_distribution():
    """Get distribution of customers by risk category"""
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        
        # Get active model
        cursor.execute("""
            SELECT model_bundle_id FROM model_registry
            WHERE is_active = TRUE
            ORDER BY training_date DESC LIMIT 1
        """)
        model_result = cursor.fetchone()
        
        if not model_result:
            raise HTTPException(status_code=404, detail="No active model found")
        
        model_id = model_result[0]
        
        # Get risk distribution
        cursor.execute("""
            SELECT risk_category, COUNT(*) as count
            FROM customer_risk_scores
            WHERE model_bundle_id = %s
            GROUP BY risk_category
        """, (model_id,))
        
        results = cursor.fetchall()
        distribution = {
            "stable_customers": 0,
            "at_risk_customers": 0,
            "high_risk_customers": 0
        }
        
        for row in results:
            category = row[0].lower().replace(' ', '_')
            distribution[f"{category}_customers"] = row[1]
        
        return distribution
    finally:
        close_connection(connection)

@router.get("/model-evolution")
async def get_model_evolution(days: int = 30):
    """Get model evolution (risk categories over time)"""
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        
        # Get summary data for each training
        cursor.execute("""
            SELECT ts.created_at, ts.stable_customers, ts.at_risk_customers, ts.high_risk_customers
            FROM training_summary ts
            JOIN model_registry mr ON ts.model_bundle_id = mr.model_bundle_id
            ORDER BY ts.created_at DESC
            LIMIT %s
        """, (days,))
        
        results = cursor.fetchall()
        evolution_data = []
        
        for row in results:
            evolution_data.append({
                "date": row[0].isoformat() if row[0] else None,
                "stable": row[1],
                "at_risk": row[2],
                "high_risk": row[3]
            })
        
        # Reverse to get chronological order
        return list(reversed(evolution_data))
    finally:
        close_connection(connection)

@router.get("/high-risk-customers")
async def get_high_risk_customers(limit: int = 10):
    """Get top high-risk customers"""
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        
        cursor.execute("""
            SELECT 
                crs.customer_id,
                crs.final_risk_score,
                crs.risk_category,
                crs.reconstruction_error,
                crs.cluster_distance,
                crs.anomaly_score,
                cf.MonthlyCharge,
                cf.AccountWeeks,
                crs.prediction_time
            FROM customer_risk_scores crs
            JOIN customer_features cf ON crs.customer_id = cf.customer_id
            JOIN model_registry mr ON crs.model_bundle_id = mr.model_bundle_id
            WHERE mr.is_active = TRUE
            ORDER BY crs.final_risk_score DESC
            LIMIT %s
        """, (limit,))
        
        results = cursor.fetchall()
        customers = []
        
        for row in results:
            customers.append({
                "customer_id": row[0],
                "final_risk_score": float(row[1]),
                "risk_category": row[2],
                "reconstruction_error": float(row[3]),
                "cluster_distance": float(row[4]),
                "anomaly_score": float(row[5]),
                "monthly_charge": float(row[6]),
                "account_weeks": row[7],
                "prediction_time": row[8].isoformat() if row[8] else None
            })
        
        return customers
    finally:
        close_connection(connection)