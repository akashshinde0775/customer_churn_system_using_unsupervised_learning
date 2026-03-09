from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from database.db_connection import get_db_connection, close_connection
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class CustomerRiskResponse(BaseModel):
    customer_id: int
    final_risk_score: float
    risk_category: str
    reconstruction_error: float
    cluster_distance: float
    anomaly_score: float
    prediction_time: str

@router.get("/{customer_id}")
async def get_customer_details(customer_id: int):
    """Get customer feature details"""
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        
        cursor.execute("""
            SELECT customer_id, AccountWeeks, ContractRenewal, DataPlan, DataUsage,
                   CustServCalls, DayMins, DayCalls, MonthlyCharge, OverageFee, RoamMins, created_at
            FROM customer_features
            WHERE customer_id = %s
        """, (customer_id,))
        
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        return {
            "customer_id": result[0],
            "AccountWeeks": result[1],
            "ContractRenewal": result[2],
            "DataPlan": result[3],
            "DataUsage": float(result[4]),
            "CustServCalls": result[5],
            "DayMins": float(result[6]),
            "DayCalls": result[7],
            "MonthlyCharge": float(result[8]),
            "OverageFee": float(result[9]),
            "RoamMins": float(result[10]),
            "created_at": result[11].isoformat() if result[11] else None
        }
    finally:
        close_connection(connection)

@router.get("/{customer_id}/risk-score")
async def get_customer_risk_score(customer_id: int):
    """Get customer risk scores and category"""
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        
        cursor.execute("""
            SELECT crs.customer_id, crs.final_risk_score, crs.risk_category,
                   crs.reconstruction_error, crs.cluster_distance, crs.anomaly_score,
                   crs.prediction_time
            FROM customer_risk_scores crs
            JOIN model_registry mr ON crs.model_bundle_id = mr.model_bundle_id
            WHERE crs.customer_id = %s AND mr.is_active = TRUE
            ORDER BY crs.prediction_time DESC
            LIMIT 1
        """, (customer_id,))
        
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="Risk score not found for this customer")
        
        return {
            "customer_id": result[0],
            "final_risk_score": float(result[1]),
            "risk_category": result[2],
            "reconstruction_error": float(result[3]),
            "cluster_distance": float(result[4]),
            "anomaly_score": float(result[5]),
            "prediction_time": result[6].isoformat() if result[6] else None
        }
    finally:
        close_connection(connection)

@router.get("/comparison")
async def compare_customers(customer_id_1: int = Query(...), customer_id_2: int = Query(...)):
    """Compare two customers' risk scores"""
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        
        # Get risk scores for both customers
        cursor.execute("""
            SELECT crs.customer_id, crs.final_risk_score, crs.risk_category,
                   crs.reconstruction_error, crs.cluster_distance, crs.anomaly_score
            FROM customer_risk_scores crs
            JOIN model_registry mr ON crs.model_bundle_id = mr.model_bundle_id
            WHERE crs.customer_id IN (%s, %s) AND mr.is_active = TRUE
            ORDER BY crs.prediction_time DESC
        """, (customer_id_1, customer_id_2))
        
        results = cursor.fetchall()
        
        if len(results) < 2:
            raise HTTPException(status_code=404, detail="Could not find both customers")
        
        comparison = []
        for row in results:
            comparison.append({
                "customer_id": row[0],
                "final_risk_score": float(row[1]),
                "risk_category": row[2],
                "reconstruction_error": float(row[3]),
                "cluster_distance": float(row[4]),
                "anomaly_score": float(row[5])
            })
        
        return comparison
    finally:
        close_connection(connection)