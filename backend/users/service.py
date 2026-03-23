# backend/users/service.py

from typing import Optional, List

from sqlalchemy.orm import Session
from passlib.hash import bcrypt  # for hashing & verifying passwords

from backend.users.models import (
    SessionLocal,
    User,
    PortfolioPosition,
)

class AccountLockedError(Exception):
    """
    Custom exception raised when a login is attempted
    on an account that has already been locked.
    """
    pass

# -------- Session helper -------- #

def get_db() -> Session:
    return SessionLocal()


# -------- User helpers -------- #

def create_user(email: str, username: str, password: str) -> User:
    """
    Create a new user with a hashed password.
    Raises ValueError if email/username already exist.
    """
    db = get_db()
    try:

        existing_email = db.query(User).filter(User.email == email).first()
        if existing_email:
            raise ValueError("A user with this email already exists.")

        existing_username = db.query(User).filter(User.username == username).first()
        if existing_username:
            raise ValueError("A user with this username already exists.")

        # 🔐 Hash the password before storing
        password_hash = bcrypt.hash(password)

        user = User(
            email=email,
            username=username,
            password=password_hash,  # we store the hash in the 'password' column
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()

MAX_FAILED_LOGIN_ATTEMPTS = 5  # keep whatever value you chose

def get_user_by_id(user_id: int) -> Optional[User]:
    """
    Load a user by primary key.
    Used for 'remember me' auto-login.
    """
    db = get_db()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        return user
    finally:
        db.close()

def authenticate_user(email: str, password: str) -> Optional[User]:
    """
    Return the user if email + password are correct, otherwise None.
    Raises AccountLockedError if the account is locked.
    """
    db = get_db()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return None

        # If the account is already locked, don't check the password
        if getattr(user, "is_locked", False):
            raise AccountLockedError(
                "This account has been locked due to too many failed login attempts."
            )

        #  Check hash instead of plain-text comparison
        try:
            if bcrypt.verify(password, user.password):
                # Successful login -> reset failed attempts counter
                user.failed_attempts = 0
                db.commit()
                #  refresh so attributes are loaded even after session closes
                db.refresh(user)
                return user
        except ValueError:
            # Hash is invalid / corrupted
            return None

        current_failed = getattr(user, "failed_attempts", 0) or 0
        current_failed += 1
        user.failed_attempts = current_failed

        if current_failed >= MAX_FAILED_LOGIN_ATTEMPTS:
            user.is_locked = True

        db.commit()
        return None
    finally:
        db.close()
        
def change_password(user_id: int, old_password: str, new_password: str) -> None:
    """
    Change a user's password after verifying the current password.
    Raises ValueError with a user-friendly message if something is wrong.
    """
    db = get_db()
    try:
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            raise ValueError("User not found.")

        # Check old password
        if not bcrypt.verify(old_password, user.password):
            raise ValueError("Current password is incorrect.")

        # Basic password policy – you can make this stricter if you want
        if len(new_password) < 8:
            raise ValueError("New password must be at least 8 characters long.")

        # Hash and store new password
        user.password = bcrypt.hash(new_password)
        db.commit()
    finally:
        db.close()

def reset_password(email: str, new_password: str) -> None:
    """
    Reset the password for the user with the given email.

    This is used by the 'Forgot password' flow in the UI.
    It does NOT send emails or tokens – it simply:
      - looks up the user by email
      - sets a new hashed password
      - raises ValueError if the email is not found
    """
    # Keep behaviour consistent with create_user / authenticate_user:
    normalized_email = email.strip()

    db = get_db()
    try:
        user: Optional[User] = (
            db.query(User)
            .filter(User.email == normalized_email)
            .first()
        )

        if user is None:
            # This gets shown in Streamlit as an error message
            raise ValueError("No account found for that email address.")

        # Basic password policy – mirror change_password at least on length
        if len(new_password) < 8:
            raise ValueError("New password must be at least 8 characters long.")

        # Hash the new password and save it
        user.password = bcrypt.hash(new_password)  # <- same column & hashing as elsewhere
        user.failed_attempts = 0
        user.is_locked = False
        db.add(user)
        db.commit()
    finally:
        db.close()


# -------- Portfolio helpers -------- #

def get_portfolio(user_id: int) -> List[PortfolioPosition]:
    db = get_db()
    try:
        positions = (
            db.query(PortfolioPosition)
            .filter(PortfolioPosition.user_id == user_id)
            .order_by(PortfolioPosition.ticker.asc())
            .all()
        )
        return positions
    finally:
        db.close()


def _find_position(db: Session, user_id: int, ticker: str) -> Optional[PortfolioPosition]:
    """
    Internal helper function to find a portfolio position
    for a given user and stock ticker.

    Parameters:
        db (Session): Active database session.
        user_id (int): ID of the user.
        ticker (str): Stock ticker symbol.

    Returns:
        Optional[PortfolioPosition]: Matching position or None.
    """
    return (
        db.query(PortfolioPosition)
        .filter(
            PortfolioPosition.user_id == user_id,
            PortfolioPosition.ticker == ticker,
        )
        .first()
    )


def buy_shares(user_id: int, ticker: str, shares: float, price: float) -> PortfolioPosition:
    """
    Basic 'buy' logic: update or create a PortfolioPosition and recompute avg_price.
    """
    db = get_db()
    try:
        pos = _find_position(db, user_id, ticker)

        if pos is None:
            pos = PortfolioPosition(
                user_id=user_id,
                ticker=ticker,
                shares=shares,
                avg_price=price,
            )
            db.add(pos)
        else:
            total_cost = pos.avg_price * pos.shares + price * shares
            total_shares = pos.shares + shares
            pos.shares = total_shares
            pos.avg_price = total_cost / total_shares

        db.commit()
        db.refresh(pos)
        return pos
    finally:
        db.close()


def sell_shares(user_id: int, ticker: str, shares: float, price: float) -> PortfolioPosition:
    """
    Basic 'sell' logic: reduce shares, keep avg_price unchanged (for simplicity).
    """
    db = get_db()
    try:
        pos = _find_position(db, user_id, ticker)

        if pos is None or pos.shares < shares:
            raise ValueError("Not enough shares to sell.")

        pos.shares -= shares

        if pos.shares == 0:
            # optionally delete position
            db.delete(pos)
            db.commit()
            return pos

        db.commit()
        db.refresh(pos)
        return pos
    finally:
        db.close()
