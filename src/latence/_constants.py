"""Constants for the Latence SDK."""

from __future__ import annotations

# API URLs
DEFAULT_BASE_URL = "https://api.latence.ai"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2

# Job polling defaults
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_POLL_TIMEOUT = 300.0

# Retry configuration defaults
DEFAULT_INITIAL_RETRY_DELAY = 0.5  # seconds
DEFAULT_MAX_RETRY_DELAY = 60.0  # seconds
DEFAULT_EXPONENTIAL_BASE = 2.0
DEFAULT_JITTER = 0.25  # 25% jitter

# Retryable HTTP status codes
RETRYABLE_STATUS_CODES = frozenset(
    {
        408,  # Request Timeout
        429,  # Too Many Requests (Rate Limited)
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    }
)

# B2 storage fetch timeouts
B2_FETCH_TIMEOUT = 60.0  # Jobs resource B2 fetch timeout
B2_PIPELINE_FETCH_TIMEOUT = 120.0  # Pipeline resource B2 fetch timeout (larger outputs)

# Presigned upload thresholds
PRESIGNED_UPLOAD_THRESHOLD = 10 * 1024 * 1024  # 10 MB -- files above this use direct B2 upload
B2_UPLOAD_TIMEOUT = 600.0  # Timeout for direct-to-B2 uploads via presigned URL
