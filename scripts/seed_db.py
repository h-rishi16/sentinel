import logging
import sys

from sentinel.data.generator import (
    GeneratorConfig,
    generate_borrowers,
    generate_loans,
    generate_transactions,
)
from sentinel.db.database import Base, SessionLocal, engine
from sentinel.db.models import Borrower, Loan, Transaction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_database():
    logger.info("Creating database tables...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    logger.info("Generating synthetic data using Sentinel Quantitative Engines...")

    n_b = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    n_l = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    n_t = int(sys.argv[3]) if len(sys.argv) > 3 else 20000

    cfg = GeneratorConfig()
    cfg.n_borrowers = n_b
    cfg.n_loans = n_l
    cfg.n_transactions = n_t
    cfg.seed = 42

    # 1. Generate core data using the Quant Data Engines
    df_borrowers = generate_borrowers(cfg)
    df_loans = generate_loans(df_borrowers, cfg)
    df_txns = generate_transactions(df_borrowers, cfg)

    # 2. Ingest into Sentinel Data Warehouse (SQLite)
    db = SessionLocal()
    try:
        logger.info("Ingesting Borrowers...")
        borrowers_records = df_borrowers.to_dict(orient="records")
        db.bulk_insert_mappings(Borrower, borrowers_records)

        logger.info("Ingesting Loans...")
        if "default" in df_loans.columns:
            df_loans = df_loans.rename(columns={"default": "is_default"})
        loans_records = df_loans.to_dict(orient="records")
        db.bulk_insert_mappings(Loan, loans_records)

        logger.info("Ingesting Transactions...")
        txns_records = df_txns.to_dict(orient="records")
        db.bulk_insert_mappings(Transaction, txns_records)

        db.commit()
        logger.info("Sentinel Data Warehouse successfully seeded!")

    except Exception as e:
        db.rollback()
        logger.error(f"Error seeding database: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
