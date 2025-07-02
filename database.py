# Tender-Aggregator-main/database.py

import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, ForeignKey, JSON, Boolean
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.sql import func
from pathlib import Path
import logging
import bcrypt
# --- Setup basic logging for database interactions ---
db_logger = logging.getLogger("database_module")
if not db_logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] (Database) %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    db_logger.addHandler(handler)
    db_logger.setLevel(logging.INFO)

# --- Configuration ---
try:
    PROJECT_ROOT = Path(__file__).parent.resolve()
except NameError:
    PROJECT_ROOT = Path('.').resolve()

DB_NAME = "tender_aggregator.db"
DATABASE_FILE_PATH = PROJECT_ROOT / DB_NAME
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DATABASE_FILE_PATH}"

# --- SQLAlchemy Setup ---
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}, # Required for SQLite
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- ORM Model Definitions ---

class Tender(Base):
    __tablename__ = "tenders"

    id = Column(Integer, primary_key=True, index=True)
    tender_id = Column(String, unique=True, index=True, nullable=False)
    source_site = Column(String, index=True, nullable=False)
    
    # Core Details
    tender_title = Column(Text)
    organisation_chain = Column(Text)
    
    # Financials
    tender_value_numeric = Column(Float, nullable=True)
    emd_amount_numeric = Column(Float, nullable=True)
    
    # Dates
    published_date = Column(DateTime, nullable=True)
    closing_date = Column(DateTime, nullable=True)
    opening_date = Column(DateTime, nullable=True)
    
    # Location
    location = Column(String, nullable=True)
    pincode = Column(String, nullable=True)
    
    # Status and Linking
    status = Column(String, default="Live", index=True)
    
    # Full data dump
    full_details_json = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    result = relationship("TenderResult", back_populates="tender", uselist=False, cascade="all, delete-orphan")
    bids = relationship("TenderBid", back_populates="tender", cascade="all, delete-orphan")
    eligibility_check = relationship("EligibilityCheck", back_populates="tender", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Tender(tender_id='{self.tender_id}', source_site='{self.source_site}', status='{self.status}')>"


class TenderResult(Base):
    __tablename__ = "tender_results"

    id = Column(Integer, primary_key=True, index=True)
    tender_id_fk = Column(Integer, ForeignKey("tenders.id"), nullable=False, unique=True)
    
    final_stage = Column(String, nullable=True)
    award_date = Column(DateTime, nullable=True)
    
    full_summary_json = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    tender = relationship("Tender", back_populates="result")

    def __repr__(self):
        return f"<TenderResult(tender.tender_id='{self.tender.tender_id if self.tender else 'N/A'}', final_stage='{self.final_stage}')>"


class CanonicalBidder(Base):
    __tablename__ = "canonical_bidders"
    __table_args__ = {'extend_existing': True} # Keep this for hot-reloading

    id = Column(Integer, primary_key=True, index=True)
    canonical_name = Column(String, unique=True, index=True, nullable=False)
    notes = Column(Text, nullable=True)
    
    aliases = relationship("Bidder", back_populates="canonical_bidder")

    def __repr__(self):
        return f"<CanonicalBidder(name='{self.canonical_name}')>"


# --- CORRECTED BIDDER CLASS ---
class Bidder(Base):
    __tablename__ = "bidders"
    __table_args__ = {'extend_existing': True} # Keep this for hot-reloading

    id = Column(Integer, primary_key=True, index=True)
    bidder_name = Column(String, unique=True, index=True, nullable=False)
    
    # Foreign Key to the canonical bidder
    canonical_id = Column(Integer, ForeignKey("canonical_bidders.id"), nullable=True)
    
    # Relationship to link back to the master record
    canonical_bidder = relationship("CanonicalBidder", back_populates="aliases")

    # Optional fields for future use
    first_seen_on_site = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Relationship to bids
    bids = relationship("TenderBid", back_populates="bidder", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Bidder(id={self.id}, name='{self.bidder_name}')>"
# --- END CORRECTION ---

# In Tender-Aggregator-main/database.py
import bcrypt # Add this import at the top

# ... (other model definitions) ...

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="user", nullable=False) # 'user' or 'admin'
    is_active = Column(Boolean, default=True)

    def set_password(self, password: str):
        # Hash the password with a salt
        pwd_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        self.hashed_password = bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')

    def check_password(self, password: str) -> bool:
        # Check a plaintext password against the stored hash
        password_bytes = password.encode('utf-8')
        hashed_password_bytes = self.hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_password_bytes)

# Don't forget to add 'bcrypt' to your requirements.txt

class TenderBid(Base):
    __tablename__ = "tender_bids"
    
    id = Column(Integer, primary_key=True, index=True)
    tender_id_fk = Column(Integer, ForeignKey("tenders.id"), nullable=False)
    bidder_id_fk = Column(Integer, ForeignKey("bidders.id"), nullable=False)
    
    bid_status = Column(String, nullable=True)
    bid_rank = Column(String, nullable=True)
    bid_value = Column(Float, nullable=True)
    
    # Relationships
    tender = relationship("Tender", back_populates="bids")
    bidder = relationship("Bidder", back_populates="bids")
    
    def __repr__(self):
        return f"<TenderBid(tender_id='{self.tender.tender_id if self.tender else 'N/A'}', bidder='{self.bidder.bidder_name if self.bidder else 'N/A'}', rank='{self.bid_rank}')>"


class CompanyProfile(Base):
    __tablename__ = "company_profile"

    id = Column(Integer, primary_key=True, index=True)
    profile_name = Column(String, unique=True, default="Default Profile")
    profile_data = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<CompanyProfile(name='{self.profile_name}')>"

class EligibilityCheck(Base):
    __tablename__ = "eligibility_checks"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    tender_id_fk = Column(Integer, ForeignKey("tenders.id"), nullable=False, unique=True)
    status = Column(String, default="pending") # pending, processing, complete, failed
    eligibility_score = Column(Integer, default=-1)
    analysis_result_json = Column(JSON, nullable=True)
    checked_at = Column(DateTime(timezone=True), server_default=func.now())
    
    tender = relationship("Tender", back_populates="eligibility_check")


def init_db():
    """
    Initializes the database and creates tables if they don't exist.
    """
    db_logger.info(f"Initializing database at: {DATABASE_FILE_PATH}")
    if not DATABASE_FILE_PATH.exists():
        db_logger.info("Database file not found, creating a new one...")
    else:
        db_logger.info("Database file already exists.")
    
    try:
        Base.metadata.create_all(bind=engine)
        db_logger.info("Tables created/verified successfully.")
        return True
    except Exception as e:
        db_logger.critical(f"Failed to create database tables: {e}", exc_info=True)
        return False

# --- Standalone Execution ---
if __name__ == "__main__":
    print("Running database initialization directly...")
    if init_db():
        print(f"Database '{DB_NAME}' created/verified successfully at '{PROJECT_ROOT}'.")
    else:
        print(f"ERROR: Database initialization failed. Check logs.")
