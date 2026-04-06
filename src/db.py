import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "inventory.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

engine = create_engine(DATABASE_URL)
Base = declarative_base()


class Inventory(Base):
    __tablename__ = "inventory"

    sku = Column(String, primary_key=True)
    current_stock = Column(Integer, default=0)
    in_transit = Column(Integer, default=0)
    lead_time_days = Column(Integer, default=7)
    safety_stock = Column(Integer, default=10)
    last_updated = Column(DateTime, default=datetime.utcnow)


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sku = Column(String)
    order_qty = Column(Integer)
    order_date = Column(DateTime)
    restock_date = Column(DateTime)
    received = Column(Integer, default=0)


Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)
