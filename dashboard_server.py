# Serve React dashboard static files from /dashboard/build if present
import os
from fastapi.staticfiles import StaticFiles
from api import app  # Import the FastAPI app instance

if os.path.isdir(os.path.join(os.path.dirname(__file__), "dashboard", "build")):
    app.mount("/dashboard", StaticFiles(directory="dashboard/build", html=True), name="dashboard")
