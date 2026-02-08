"""
Small CLI smoke-test script for the users & portfolio layer.

Run from the project root with:

    python -m backend.users.test_users

This will:
  * Ensure the DB tables exist
  * Create a user if it doesn't exist yet
  * Run a few buy / sell operations
  * Print the resulting portfolio
"""

from getpass import getpass

from backend.users.models import init_db
from backend.users.service import (
    create_user,
    authenticate_user,
    buy_shares,
    sell_shares,
    get_portfolio,
)


def main() -> None:
    # Ensure DB tables exist
    init_db()

    print("=== User / portfolio smoke test ===")
    print("If the user already exists, we'll just log in.\n")

    email = input("Email for test user: ").strip()
    if not email:
        print("Email is required, aborting.")
        return

    username = input("Username (if creating new user): ").strip()
    if not username:
        # Fallback: derive from email local-part
        username = email.split("@")[0]

    password = getpass("Password: ").strip()
    if not password:
        print("Password is required, aborting.")
        return

    # 1. Try to create user (or fall back to authenticate)
    try:
        user = create_user(
            email=email,
            username=username,
            password=password,
        )
        print(f"Created user: {user.id} - {user.username}")
    except ValueError as e:
        print(f"User creation skipped: {e}")
        user = authenticate_user(email=email, password=password)
        if not user:
            print("Failed to authenticate existing user – check password.")
            return
        print(f"Authenticated existing user: {user.id} - {user.username}")

    # 2. Run some simple portfolio operations
    print("\n--- Running portfolio operations for", user.username, "---")

    # Buy 1 share
    pos = buy_shares(user_id=user.id, ticker="AAPL", shares=1, price=180.0)
    print(f"After first buy: {pos.ticker}, shares={pos.shares}, avg_price={pos.avg_price:.2f}")

    # Buy more
    pos = buy_shares(user_id=user.id, ticker="AAPL", shares=2, price=190.0)
    print(f"After second buy: {pos.ticker}, shares={pos.shares}, avg_price={pos.avg_price:.2f}")

    # Sell some
    pos = sell_shares(user_id=user.id, ticker="AAPL", shares=1, price=200.0)
    print(f"After sell: {pos.ticker}, shares={pos.shares}, avg_price={pos.avg_price:.2f}")

    # 3. Show final portfolio
    print("\nCurrent portfolio:")
    portfolio = get_portfolio(user.id)
    if not portfolio:
        print("  (empty)")
    else:
        for p in portfolio:
            print(f"  {p.ticker}: {p.shares} shares @ {p.avg_price:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
