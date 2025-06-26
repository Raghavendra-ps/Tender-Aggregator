# Tender-Aggregator-main/database.py

import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, ForeignKey, JSON
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.sql import func
from pathlib import Path
import logging

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
    # echo=True # Uncomment for debugging SQL queries
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
    status = Column(String, default="Live", index=True) # e.g., "Live", "Result Announced", "Cancelled"
    
    # Full data dump
    full_details_json = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship to TenderResult
    result = relationship("TenderResult", back_populates="tender", uselist=False, cascade="all, delete-orphan")
    bids = relationship("TenderBid", back_populates="tender", cascade="all, delete-orphan")

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
    
    # Relationship back to Tender
    tender = relationship("Tender", back_populates="result")

    def __repr__(self):
        return f"<TenderResult(tender_id='{self.tender.tender_id}', final_stage='{self.final_stage}')>"


class Bidder(Base):
    __tablename__ = "bidders"
    
    id = Column(Integer, primary_key=True, index=True)
    bidder_name = Column(String, unique=True, index=True, nullable=False)
    
    # Optional fields for future use
    first_seen_on_site = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Relationship to bids
    bids = relationship("TenderBid", back_populates="bidder", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Bidder(id={self.id}, name='{self.bidder_name}')>"


class TenderBid(Base):
    __tablename__ = "tender_bids"
    
    id = Column(Integer, primary_key=True, index=True)
    tender_id_fk = Column(Integer, ForeignKey("tenders.id"), nullable=False)
    bidder_id_fk = Column(Integer, ForeignKey("bidders.id"), nullable=False)
    
    bid_status = Column(String, nullable=True) # e.g., "Accepted-Finance", "Rejected-Technical"
    bid_rank = Column(String, nullable=True) # e.g., "L1", "L2"
    bid_value = Column(Float, nullable=True)
    
    # Relationships
    tender = relationship("Tender", back_populates="bids")
    bidder = relationship("Bidder", back_populates="bids")
    
    def __repr__(self):
        return f"<TenderBid(tender_id='{self.tender.tender_id}', bidder='{self.bidder.bidder_name}', rank='{self.bid_rank}')>"


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
