import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Float, JSON, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@db:5432/quarry")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True)
    status = Column(String, nullable=False, default="pending")
    progress = Column(Integer, default=0)
    input_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    error = Column(Text, nullable=True)
    result_count = Column(Integer, default=0)


class Site(Base):
    __tablename__ = "sites"

    id = Column(String, primary_key=True)
    site_id = Column(String, nullable=False, index=True, unique=True)
    job_id = Column(String, nullable=False, index=True)
    data = Column(JSON, nullable=False)
    official_name = Column(String, nullable=True)
    operational_status = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    return SessionLocal()
