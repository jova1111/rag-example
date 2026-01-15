"""Database connection management."""

import psycopg2
from psycopg2.extras import RealDictCursor
from config import Config


class DatabaseConnection:
    """Manage PostgreSQL database connections."""
    
    def __init__(self):
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(
                Config.get_db_connection_string(),
                cursor_factory=RealDictCursor
            )
            self.cursor = self.conn.cursor()
            # Register pgvector extension
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.conn.commit()
            print("✓ Connected to PostgreSQL database")
            return self.conn, self.cursor
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            if self.conn:
                self.conn.rollback()
        else:
            if self.conn:
                self.conn.commit()
        self.close()


def get_connection():
    """Get a new database connection."""
    db = DatabaseConnection()
    db.connect()
    return db.conn, db.cursor
