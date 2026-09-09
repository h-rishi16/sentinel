from sqlalchemy import Column, Integer, Float, String, Boolean, ForeignKey
from sentinel.db.database import Base

class Borrower(Base):
    __tablename__ = "borrowers"
    id = Column(Integer, primary_key=True, index=True)
    borrower_id = Column(Integer, unique=True, index=True)
    annual_income = Column(Float)
    fico_score = Column(Integer)
    emp_length_years = Column(Integer)

class Loan(Base):
    __tablename__ = "loans"
    id = Column(Integer, primary_key=True, index=True)
    loan_id = Column(Integer, unique=True, index=True)
    borrower_id = Column(Integer, ForeignKey("borrowers.borrower_id"))
    loan_amount = Column(Float)
    term_months = Column(Integer)
    interest_rate = Column(Float)
    dti = Column(Float)
    is_secured = Column(Integer)
    recovery_rate = Column(Float)
    lgd = Column(Float)
    ead_factor = Column(Float)
    ead = Column(Float)
    is_default = Column(Integer)

class Transaction(Base):
    __tablename__ = "transactions"
    txn_id = Column(Integer, primary_key=True, index=True)
    card_id = Column(Integer, ForeignKey("borrowers.borrower_id"))
    amount = Column(Float)
    category = Column(String)
    day = Column(Integer)
    hour = Column(Integer)
    is_fraud = Column(Integer)
