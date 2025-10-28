"""
seed_db.py
Insert parquet data into Postgres table named by --table.
Usage:
  python seed_db.py --parquet data/ag_news_50000.parquet --table hf_docs_50k
"""
import argparse
import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text, Column, Integer, Text, TIMESTAMP, func
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.types import UserDefinedType

# Custom Vector UDT
class Vector(UserDefinedType):
    def __init__(self, dimensions):
        self.dimensions = dimensions
    def get_col_spec(self):
        return f"vector({self.dimensions})"
    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            return "[" + ",".join(str(float(v)) for v in value) + "]"
        return process
    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            return [float(x) for x in value.strip("[]").split(",")]
        return process

parser = argparse.ArgumentParser()
parser.add_argument("--parquet", required=True)
parser.add_argument("--table", required=True)
parser.add_argument("--dim", type=int, default=384)
args = parser.parse_args()

load_dotenv()  # read .env in cwd

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError("Missing DB credentials in .env")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL, future=True)

Base = declarative_base()

class HFDocs(Base):
    __tablename__ = args.table
    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    label = Column(Integer)
    text_len = Column(Integer)
    ts = Column(TIMESTAMP(timezone=True), server_default=func.now())
    embedding = Column(Vector(args.dim))

print("Loading parquet:", args.parquet)
df = pd.read_parquet(args.parquet)
df["embedding"] = df["embedding"].apply(lambda v: [float(x) for x in v])

with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    conn.commit()

# Drop/create table fresh (dev)
Base.metadata.drop_all(bind=engine, checkfirst=True)
Base.metadata.create_all(bind=engine)
print(f"Created table {args.table}")

# Bulk insert
with Session(engine) as session:
    objs = []
    for _, row in df.iterrows():
        objs.append(HFDocs(
            id=int(row["id"]),
            text=str(row["text"]),
            label=int(row["label"]) if row["label"] is not None else None,
            text_len=int(row["text_len"]),
            ts=row["ts"],
            embedding=row["embedding"]
        ))
    session.bulk_save_objects(objs)
    session.commit()
print(f"Inserted {len(df)} rows into {args.table}")
