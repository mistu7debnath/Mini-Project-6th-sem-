from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import rewrite

# ✅ FIRST create app
app = FastAPI()

# ✅ THEN add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ THEN include routes
app.include_router(rewrite.router)