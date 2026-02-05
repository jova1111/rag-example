"""Database connection management."""

import psycopg2
import asyncpg
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


# Global connection pool
_pool = None


async def get_pool():
    """Get or create the global connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            min_size=10,  # Minimum number of connections
            max_size=100,  # Maximum number of connections
            command_timeout=60
        )
        print("✓ Created async database connection pool")
    return _pool


async def close_pool():
    """Close the global connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        print("✓ Closed async database connection pool")


class AsyncDatabaseConnection:
    """Manage async PostgreSQL database connections using connection pool."""
    
    def __init__(self):
        self.conn = None
        self.pool = None
    
    async def connect(self):
        """Acquire connection from pool."""
        try:
            self.pool = await get_pool()
            self.conn = await self.pool.acquire()
            
            # Register pgvector extension (only needed once, but safe to run multiple times)
            await self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            return self.conn
        except Exception as e:
            print(f"✗ Async database connection failed: {e}")
            raise
    
    async def close(self):
        """Release connection back to pool."""
        if self.conn and self.pool:
            await self.pool.release(self.conn)
            self.conn = None
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


async def get_async_connection():
    """Get a new async database connection from pool."""
    pool = await get_pool()
    return await pool.acquire()
