from fastapi import APIRouter, Request

from app.usecases.global_config import (
    get_logo_path,
    get_available_models_for_user,
    get_default_model,
)

router = APIRouter(tags=["config"])


@router.get("/config/global")
def get_global_config(request: Request):
    """Get global configuration including available models."""
    current_user = getattr(request.state, "current_user", None)
    global_models = get_available_models_for_user(current_user)
    default_model = get_default_model()
    logo_path = get_logo_path()
    return {
        "globalAvailableModels": global_models,
        "defaultModel": default_model,
        "logoPath": logo_path,
    }
