import json
import logging
import os

from app.routes.schemas.conversation import type_model_name
from app.user import User

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s - %(message)s")
logger = logging.getLogger(__name__)

GLOBAL_AVAILABLE_MODELS = os.environ.get("GLOBAL_AVAILABLE_MODELS")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL")
TITLE_MODEL = os.environ.get("TITLE_MODEL")
LOGO_PATH = os.environ.get("LOGO_PATH", "")


def get_global_available_models() -> list[str]:
    """
    Get the list of globally available models from environment variable.
    Returns empty list if not configured or if an empty array is provided,
    which means all models are available.
    """
    if GLOBAL_AVAILABLE_MODELS:
        try:
            models = json.loads(GLOBAL_AVAILABLE_MODELS)
            # Ensure the result is a list
            if isinstance(models, list):
                logger.info(f"Global available models (JSON): {models}")
                # Filter out empty strings and None values
                filtered_models = [
                    model for model in models if model and isinstance(model, str)
                ]
                return filtered_models
            else:
                logger.error(
                    f"GLOBAL_AVAILABLE_MODELS must be a JSON array, got {type(models)}"
                )
                return []
        except json.JSONDecodeError:
            logger.error("Failed to parse GLOBAL_AVAILABLE_MODELS as JSON")
            return []

    logger.info("No global available models configured - all models are available")
    return []


def get_available_models_for_user(user: User | None) -> list[str]:
    """Return the model allowlist for a user.

    An empty list is the existing application convention for unrestricted model
    access. Only authenticated administrators receive that unrestricted view;
    all other callers receive the configured employee allowlist.
    """
    if user is not None and user.is_admin():
        logger.info("Admin user has unrestricted model access")
        return []

    return get_global_available_models()


def ensure_model_available_for_user(user: User, model: str) -> None:
    """Reject attempts to use a model outside the user's server-side allowlist."""
    available_models = get_available_models_for_user(user)
    if available_models and model not in available_models:
        raise PermissionError("The selected model is not available for this user.")


def get_default_model() -> type_model_name:
    """Return the configured default model, falling back to claude-v3.7-sonnet."""
    return (DEFAULT_MODEL or "").strip() or "claude-v3.7-sonnet"  # type: ignore[return-value]


def get_title_model() -> type_model_name:
    """Return the model for title generation.

    Falls back to: TITLE_MODEL -> DEFAULT_MODEL -> claude-v3-haiku
    """
    return (TITLE_MODEL or "").strip() or (DEFAULT_MODEL or "").strip() or "claude-v3-haiku"  # type: ignore[return-value]


def get_logo_path() -> str:
    """Return the configured drawer logo path."""
    return LOGO_PATH
