import os
import datetime
import json
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, String, Float, LargeBinary, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Flag to track database availability
DB_AVAILABLE = True

# Base class for models
Base = declarative_base()

class AnalysisResult(Base):
    """Model for storing analysis results."""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(50))
    entropy_value = Column(Float)
    meta_data = Column(Text)  # JSON string - renamed from 'metadata' which is reserved
    analysis_date = Column(DateTime, default=datetime.datetime.utcnow)
    thumbnail = Column(LargeBinary, nullable=True)  # For image files
    
    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, filename='{self.filename}')>"

# Initialize engine and session factory
engine = None
SessionLocal = None

# Try to initialize database connection
try:
    # Get the database URL from environment variables
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    if DATABASE_URL:
        # Create engine with connection pooling and SSL handling
        engine = create_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,  # Recycle connections every 30 minutes
            connect_args={
                "sslmode": "prefer",
                "connect_timeout": 10
            }
        )
        
        # Try to create tables
        Base.metadata.create_all(engine)
        
        # Session factory
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    else:
        print("WARNING: DATABASE_URL environment variable not set")
        DB_AVAILABLE = False
except Exception as e:
    print(f"ERROR: Failed to initialize database: {str(e)}")
    DB_AVAILABLE = False

def get_db_session():
    """Get a database session."""
    if not DB_AVAILABLE or not SessionLocal:
        return None
        
    try:
        session = SessionLocal()
        return session
    except Exception as e:
        print(f"Failed to create database session: {str(e)}")
        return None

def save_analysis(filename, file_size, file_type, entropy_value, metadata, thumbnail=None):
    """Save analysis results to database with robust error handling."""
    global DB_AVAILABLE
    
    if not DB_AVAILABLE:
        return None
        
    session = None
    try:
        session = get_db_session()
        if not session:
            return None
            
        # Convert NumPy types to Python native types to avoid database errors
        if hasattr(entropy_value, 'item'):  # Check if it's a NumPy type
            entropy_value = float(entropy_value.item())  # Convert to Python float
        else:
            entropy_value = float(entropy_value)  # Ensure it's a float
            
        analysis = AnalysisResult(
            filename=filename,
            file_size=file_size,
            file_type=file_type,
            entropy_value=entropy_value,
            meta_data=metadata,
            thumbnail=thumbnail
        )
        session.add(analysis)
        session.commit()
        return analysis.id
    except Exception as e:
        # Silently handle database errors - don't spam user with DB issues
        if session:
            try:
                session.rollback()
            except:
                pass
        DB_AVAILABLE = False  # Disable database for rest of session if failing
        return None
    finally:
        if session:
            session.close()

def get_recent_analyses(limit=10):
    """Get recent analysis records."""
    if not DB_AVAILABLE:
        print("Database not available, returning empty results")
        return []
        
    session = get_db_session()
    if not session:
        return []
        
    try:
        results = session.query(AnalysisResult).order_by(
            AnalysisResult.analysis_date.desc()
        ).limit(limit).all()
        return results
    except Exception as e:
        print(f"Error getting recent analyses: {str(e)}")
        return []
    finally:
        if session:
            session.close()

def get_analysis_by_id(analysis_id):
    """Get analysis by ID."""
    if not DB_AVAILABLE:
        print("Database not available, returning None")
        return None
        
    session = get_db_session()
    if not session:
        return None
        
    try:
        result = session.query(AnalysisResult).filter(
            AnalysisResult.id == analysis_id
        ).first()
        return result
    except Exception as e:
        print(f"Error getting analysis by ID: {str(e)}")
        return None
    finally:
        if session:
            session.close()