# Tender-Aggregator-main/auth.py

from datetime import datetime, timedelta, timezone
from typing import Optional
import os

# --- THIS IS THE FIX ---
from fastapi import Depends, HTTPException, status, Request, Response
from fastapi.responses import RedirectResponse # Import RedirectResponse
# --- END FIX ---

from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from database import SessionLocal, User

# --- Configuration ---
# IMPORTANT: Generate a new key and store it securely (e.g., environment variable)
# Command: openssl rand -hex 32
SECRET_KEY = os.environ.get("SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- Dependency Functions ---

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

async def get_current_user_from_cookie(request: Request) -> Optional[User]:
    token = request.cookies.get("access_token")
    if not token:
        return None
        
    if token.startswith("Bearer "):
        token = token.split("Bearer ")[1]
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        token_data = TokenData(username=username, role=payload.get("role"))
    except JWTError:
        return None
    
    db = SessionLocal()
    user = db.query(User).filter(User.username == token_data.username).first()
    db.close()
    
    if user and user.is_active:
        user.role = token_data.role # Attach role for convenience
        return user
    return None

def require_user(user: Optional[User] = Depends(get_current_user_from_cookie)):
    """Dependency to protect routes that require any logged-in user."""
    if not user:
        # If no user is found, redirect them to the login page.
        return RedirectResponse(url="/login", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
    return user

def require_admin(user: Optional[User] = Depends(get_current_user_from_cookie)):
    """Dependency to protect routes that require an admin user."""
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access required")
    return user
