"""
Enterprise Commercial FastAPI REST API Microservice for ML-Journey.
Exposes JSON REST endpoints for Credit Risk Scoring, Telemarketing Campaign ROI Optimization, and Churn Retention Offers.
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field

from ml_journey.risk_scoring import probability_to_credit_score, get_credit_risk_tier
from ml_journey.financial import compute_financial_profit, find_optimal_profit_threshold
from ml_journey.churn import calculate_customer_ltv, calculate_max_retention_discount
from ml_journey.api_auth import verify_api_key, VALID_API_KEYS

# Initialize FastAPI App
app = FastAPI(
    title="ML-Journey Enterprise Intelligence API 🚀",
    description="Commercial REST API Microservice serving Credit Risk Scores, Financial ROI Campaign Thresholds, and Churn LTV Retention Offers.",
    version="1.0.0",
)

# API Key Dependency Middleware
def get_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    if not verify_api_key(x_api_key):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized API Key. Provide a valid 'X-API-Key' header.",
        )
    return x_api_key


# --- Pydantic Data Models ---

class CreditRiskRequest(BaseModel):
    credit_limit: float = Field(..., example=50000.0, description="Approved credit limit ($)")
    payment_delay_status: int = Field(..., example=1, description="Past payment delay months (0 = On time, 1 = 1 month delay, etc.)")
    current_bill_amount: float = Field(..., example=25000.0, description="Current bill statement amount ($)")
    last_payment_amount: float = Field(..., example=1500.0, description="Most recent payment amount ($)")


class CreditRiskResponse(BaseModel):
    credit_score: int = Field(..., description="FICO-style Credit Risk Score (300 to 850)")
    risk_tier: str = Field(..., description="Risk Category Tier")
    estimated_default_probability: float = Field(..., description="Predicted Default Probability (0.0 to 1.0)")
    underwriting_decision: str = Field(..., description="Automated Credit Approval Recommendation")


class CampaignROIRequest(BaseModel):
    client_probabilities: List[float] = Field(..., example=[0.85, 0.45, 0.12, 0.92, 0.05], description="List of predicted conversion probabilities")
    cost_per_call: float = Field(10.0, example=10.0, description="Operational cost ($) per telemarketing call")
    revenue_per_conversion: float = Field(100.0, example=100.0, description="Revenue ($) generated per conversion")


class CampaignROIResponse(BaseModel):
    optimal_call_threshold: float = Field(..., description="Profit-maximizing decision cutoff threshold")
    max_net_profit: float = Field(..., description="Achievable net campaign profit ($)")
    total_calls_to_make: int = Field(..., description="Target number of high-probability clients to call")


class ChurnIncentiveRequest(BaseModel):
    monthly_spend: float = Field(..., example=85.0, description="Subscriber average monthly spend ($)")
    tenure_months: int = Field(..., example=18, description="Contract tenure in months")
    predicted_churn_risk: float = Field(..., example=0.72, description="Predicted churn probability (0.0 to 1.0)")


class ChurnIncentiveResponse(BaseModel):
    customer_ltv: float = Field(..., description="Calculated Customer Lifetime Value ($)")
    max_retention_discount: float = Field(..., description="Maximum profitable retention discount ($)")
    retention_priority: str = Field(..., description="Priority Status for Retention Team")


# --- REST Endpoints ---

@app.get("/", tags=["System"])
def root():
    return {
        "service": "ML-Journey Enterprise Intelligence API",
        "status": "Operational",
        "documentation": "/docs",
        "version": "1.0.0",
    }


@app.post(
    "/v1/credit-score",
    response_model=CreditRiskResponse,
    tags=["Fintech Credit Scoring"],
    summary="Compute FICO-style Credit Risk Score & Decision",
)
def compute_credit_score(
    request: CreditRiskRequest, api_key: str = Depends(get_api_key)
):
    """
    Computes standard FICO-style Credit Risk Score (300-850) and Risk Tier for underwriting.
    """
    est_prob = float(
        np.clip(
            0.05
            + (request.payment_delay_status * 0.22)
            + ((request.current_bill_amount - request.last_payment_amount) / (request.credit_limit + 1e-5)) * 0.15,
            0.01,
            0.95,
        )
    )

    score = probability_to_credit_score(est_prob)
    tier_label, _ = get_credit_risk_tier(score)
    decision = "APPROVED" if score >= 670 else "REJECTED (High Default Risk)"

    return {
        "credit_score": score,
        "risk_tier": tier_label,
        "estimated_default_probability": round(est_prob, 4),
        "underwriting_decision": decision,
    }


@app.post(
    "/v1/campaign-roi",
    response_model=CampaignROIResponse,
    tags=["Marketing ROI Optimization"],
    summary="Calculate Profit-Maximizing Call Cutoff Threshold",
)
def optimize_campaign_roi(
    request: CampaignROIRequest, api_key: str = Depends(get_api_key)
):
    """
    Calculates exact decision threshold cutoff to maximize marketing campaign Net Profit ($).
    """
    probs = np.array(request.client_probabilities)
    mock_true = (probs >= 0.5).astype(int)

    best_th, max_p, _ = find_optimal_profit_threshold(
        mock_true, probs, cost_per_call=request.cost_per_call, value_per_conversion=request.revenue_per_conversion
    )

    target_calls = int(np.sum(probs >= best_th))

    return {
        "optimal_call_threshold": round(best_th, 4),
        "max_net_profit": round(max_p, 2),
        "total_calls_to_make": target_calls,
    }


@app.post(
    "/v1/churn-incentive",
    response_model=ChurnIncentiveResponse,
    tags=["Subscriber Retention"],
    summary="Calculate Optimal Retention Incentive Offer",
)
def compute_churn_incentive(
    request: ChurnIncentiveRequest, api_key: str = Depends(get_api_key)
):
    """
    Calculates Customer Lifetime Value (LTV) and maximum profitable retention discount offer ($).
    """
    ltv = calculate_customer_ltv(request.monthly_spend, request.tenure_months)
    ret_res = calculate_max_retention_discount(request.predicted_churn_risk, ltv)

    return {
        "customer_ltv": round(ltv, 2),
        "max_retention_discount": round(ret_res["Max Retention Discount ($)"], 2),
        "retention_priority": ret_res["Retention Priority"],
    }
