from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from core.config import get_settings
from core.exceptions import PipelineBaseError
from core.logging import configure_logging, get_logger
from queues.connection import get_redis_connection
from app.routes import health, webhook
from fastapi import Request
from fastapi.responses import JSONResponse

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        conn = get_redis_connection()
        conn.ping()
        logger.info("redis_connected", url=settings.redis.url)
    except Exception as exc:
        logger.warning("redis_unavailable", error=str(exc))
    yield


app = FastAPI(
    title="Survey Report Generation API",
    version="1.0.0",
    description="Receives FileMaker webhooks and processes survey reports via AI pipeline.",
    lifespan=lifespan,
)

# ── CORS — allow FileMaker and other clients to call the API ─────────
_settings = get_settings()
_origins = (
    ["*"]
    if _settings.allowed_origins.strip() == "*"
    else [o.strip() for o in _settings.allowed_origins.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    settings = get_settings()

    # Allow health + preflight
    if request.method == "OPTIONS" or request.url.path.startswith("/health"):
        return await call_next(request)

    api_key = request.headers.get("x-api-key")

    if not api_key or api_key != settings.webhook_api_key:
        logger.warning(
            "unauthorized_request",
            path=request.url.path,
            client=request.client.host if request.client else None,
        )
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"},
        )

    return await call_next(request)

@app.exception_handler(PipelineBaseError)
async def pipeline_error_handler(request, exc: PipelineBaseError):
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(webhook.router, prefix="/webhook", tags=["webhook"])
