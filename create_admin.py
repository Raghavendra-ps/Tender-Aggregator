# In Tender-Aggregator-main/create_admin.py
import sys
from getpass import getpass
from database import SessionLocal, User, init_db

def create_admin_user():
    print("--- Create Admin User ---")
    
    # Initialize DB if it doesn't exist
    init_db()
    
    db = SessionLocal()
    
    try:
        username = input("Enter admin username: ").strip()
        if not username:
            print("Username cannot be empty.")
            return

        if db.query(User).filter(User.username == username).first():
            print(f"User '{username}' already exists.")
            return
            
        password = getpass("Enter admin password: ")
        if not password:
            print("Password cannot be empty.")
            return
            
        confirm_password = getpass("Confirm admin password: ")
        if password != confirm_password:
            print("Passwords do not match.")
            return

        admin_user = User(username=username, role="admin")
        admin_user.set_password(password)
        
        db.add(admin_user)
        db.commit()
        
        print(f"\n✅ Admin user '{username}' created successfully!")
        
    finally:
        db.close()

if __name__ == "__main__":
    create_admin_user()
