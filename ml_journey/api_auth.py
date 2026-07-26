"""
API Key authentication and usage metering utilities for monetized microservices.
"""

from typing import Dict, Optional


# Mock database of authorized client API keys and billing tiers
VALID_API_KEYS: Dict[str, Dict[str, str]] = {
    "sk_live_fintech_78a9b2c": {"client_name": "Acme Credit Union", "tier": "Enterprise", "quota_remaining": "998,420"},
    "sk_live_telecom_44f1c9e": {"client_name": "Global Telecom SaaS", "tier": "Pro", "quota_remaining": "48,150"},
    "sk_demo_free_123456789": {"client_name": "Demo Sandbox Developer", "tier": "Free Sandbox", "quota_remaining": "100"},
}


def verify_api_key(api_key: Optional[str]) -> bool:
    """
    Validates client API key against authorized database.

    Args:
        api_key: Client provided API key string.

    Returns:
        is_valid: True if API key is authorized.
    """
    if not api_key:
        return False
    return api_key in VALID_API_KEYS
