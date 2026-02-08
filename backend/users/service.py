# backend/users/service.py

from typing import Optional, List

from sqlalchemy.orm import Session
from passlib.hash import bcrypt  # 👈 for hashing & verifying passwords

from backend.users.models import (
    SessionLocal,
    User,
    PortfolioPosition,
)

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


def authenticate_user(email: str, password: str) -> Optional[User]:
    """
    Return the user if email + password are correct, otherwise None.
    """
    db = get_db()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return None

        # 🔐 Check hash instead of plain-text comparison
        try:
            if bcrypt.verify(password, user.password):
                return user
        except ValueError:
            # hash is invalid / corrupted
            return None

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
