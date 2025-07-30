import sqlite3
import psycopg2
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os


@dataclass
class CardRecord:
    """Data class for Pokemon card records"""
    id: Optional[int] = None
    card_name: str = ""
    image_path: str = ""
    last_price: Optional[float] = None
    last_checked: Optional[datetime] = None


class DatabaseInterface(ABC):
    """Abstract base class for database operations"""
    
    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def disconnect(self):
        pass
    
    @abstractmethod
    def create_tables(self):
        pass
    
    @abstractmethod
    def insert_card(self, card: CardRecord) -> int:
        pass
    
    @abstractmethod
    def update_card_price(self, card_id: int, price: float, timestamp: datetime = None) -> bool:
        pass
    
    @abstractmethod
    def get_card_by_name(self, card_name: str) -> Optional[CardRecord]:
        pass
    
    @abstractmethod
    def get_card_by_id(self, card_id: int) -> Optional[CardRecord]:
        pass
    
    @abstractmethod
    def get_all_cards(self) -> List[CardRecord]:
        pass
    
    @abstractmethod
    def delete_card(self, card_id: int) -> bool:
        pass


class SQLiteDatabase(DatabaseInterface):
    """SQLite implementation of the database interface"""
    
    def __init__(self, db_path: str = "pokemon_cards.db"):
        self.db_path = db_path
        self.connection = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Connect to SQLite database"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            print(f"✅ Connected to SQLite database: {self.db_path}")
        except sqlite3.Error as e:
            print(f"❌ Error connecting to SQLite: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("🔒 Database connection closed")
    
    def create_tables(self):
        """Create the pokemon_cards table"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS pokemon_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            card_name TEXT NOT NULL UNIQUE,
            image_path TEXT NOT NULL,
            last_price REAL,
            last_checked TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_card_name ON pokemon_cards(card_name);
        CREATE INDEX IF NOT EXISTS idx_last_checked ON pokemon_cards(last_checked);
        """
        
        try:
            cursor = self.connection.cursor()
            cursor.executescript(create_table_sql)
            self.connection.commit()
            print("✅ Tables created successfully")
        except sqlite3.Error as e:
            print(f"❌ Error creating tables: {e}")
            raise
    
    def insert_card(self, card: CardRecord) -> int:
        """Insert a new card record"""
        insert_sql = """
        INSERT INTO pokemon_cards (card_name, image_path, last_price, last_checked)
        VALUES (?, ?, ?, ?)
        """
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(insert_sql, (
                card.card_name,
                card.image_path,
                card.last_price,
                card.last_checked
            ))
            self.connection.commit()
            card_id = cursor.lastrowid
            print(f"✅ Card '{card.card_name}' inserted with ID: {card_id}")
            return card_id
        except sqlite3.IntegrityError:
            print(f"⚠️ Card '{card.card_name}' already exists")
            return -1
        except sqlite3.Error as e:
            print(f"❌ Error inserting card: {e}")
            raise
    
    def update_card_price(self, card_id: int, price: float, timestamp: datetime = None) -> bool:
        """Update card price and timestamp"""
        if timestamp is None:
            timestamp = datetime.now()
        
        update_sql = """
        UPDATE pokemon_cards 
        SET last_price = ?, last_checked = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(update_sql, (price, timestamp, card_id))
            self.connection.commit()
            
            if cursor.rowcount > 0:
                print(f"✅ Updated card ID {card_id} with price ${price:.2f}")
                return True
            else:
                print(f"⚠️ No card found with ID {card_id}")
                return False
        except sqlite3.Error as e:
            print(f"❌ Error updating card: {e}")
            raise
    
    def get_card_by_name(self, card_name: str) -> Optional[CardRecord]:
        """Get card by name"""
        select_sql = "SELECT * FROM pokemon_cards WHERE card_name = ?"
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(select_sql, (card_name,))
            row = cursor.fetchone()
            
            if row:
                return CardRecord(
                    id=row['id'],
                    card_name=row['card_name'],
                    image_path=row['image_path'],
                    last_price=row['last_price'],
                    last_checked=datetime.fromisoformat(row['last_checked']) if row['last_checked'] else None
                )
            return None
        except sqlite3.Error as e:
            print(f"❌ Error getting card by name: {e}")
            raise
    
    def get_card_by_id(self, card_id: int) -> Optional[CardRecord]:
        """Get card by ID"""
        select_sql = "SELECT * FROM pokemon_cards WHERE id = ?"
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(select_sql, (card_id,))
            row = cursor.fetchone()
            
            if row:
                return CardRecord(
                    id=row['id'],
                    card_name=row['card_name'],
                    image_path=row['image_path'],
                    last_price=row['last_price'],
                    last_checked=datetime.fromisoformat(row['last_checked']) if row['last_checked'] else None
                )
            return None
        except sqlite3.Error as e:
            print(f"❌ Error getting card by ID: {e}")
            raise
    
    def get_all_cards(self) -> List[CardRecord]:
        """Get all cards"""
        select_sql = "SELECT * FROM pokemon_cards ORDER BY card_name"
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(select_sql)
            rows = cursor.fetchall()
            
            cards = []
            for row in rows:
                cards.append(CardRecord(
                    id=row['id'],
                    card_name=row['card_name'],
                    image_path=row['image_path'],
                    last_price=row['last_price'],
                    last_checked=datetime.fromisoformat(row['last_checked']) if row['last_checked'] else None
                ))
            return cards
        except sqlite3.Error as e:
            print(f"❌ Error getting all cards: {e}")
            raise
    
    def delete_card(self, card_id: int) -> bool:
        """Delete a card by ID"""
        delete_sql = "DELETE FROM pokemon_cards WHERE id = ?"
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(delete_sql, (card_id,))
            self.connection.commit()
            
            if cursor.rowcount > 0:
                print(f"✅ Deleted card with ID {card_id}")
                return True
            else:
                print(f"⚠️ No card found with ID {card_id}")
                return False
        except sqlite3.Error as e:
            print(f"❌ Error deleting card: {e}")
            raise


class PostgreSQLDatabase(DatabaseInterface):
    """PostgreSQL implementation of the database interface"""
    
    def __init__(self, host: str, database: str, user: str, password: str, port: int = 5432):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.connection = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            print(f"✅ Connected to PostgreSQL database: {self.database}")
        except psycopg2.Error as e:
            print(f"❌ Error connecting to PostgreSQL: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("🔒 Database connection closed")
    
    def create_tables(self):
        """Create the pokemon_cards table"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS pokemon_cards (
            id SERIAL PRIMARY KEY,
            card_name VARCHAR(255) NOT NULL UNIQUE,
            image_path TEXT NOT NULL,
            last_price DECIMAL(10,2),
            last_checked TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_card_name ON pokemon_cards(card_name);
        CREATE INDEX IF NOT EXISTS idx_last_checked ON pokemon_cards(last_checked);
        """
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(create_table_sql)
            self.connection.commit()
            print("✅ Tables created successfully")
        except psycopg2.Error as e:
            print(f"❌ Error creating tables: {e}")
            raise
    
    def insert_card(self, card: CardRecord) -> int:
        """Insert a new card record"""
        insert_sql = """
        INSERT INTO pokemon_cards (card_name, image_path, last_price, last_checked)
        VALUES (%s, %s, %s, %s) RETURNING id
        """
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(insert_sql, (
                card.card_name,
                card.image_path,
                card.last_price,
                card.last_checked
            ))
            card_id = cursor.fetchone()[0]
            self.connection.commit()
            print(f"✅ Card '{card.card_name}' inserted with ID: {card_id}")
            return card_id
        except psycopg2.IntegrityError:
            self.connection.rollback()
            print(f"⚠️ Card '{card.card_name}' already exists")
            return -1
        except psycopg2.Error as e:
            self.connection.rollback()
            print(f"❌ Error inserting card: {e}")
            raise
    
    def update_card_price(self, card_id: int, price: float, timestamp: datetime = None) -> bool:
        """Update card price and timestamp"""
        if timestamp is None:
            timestamp = datetime.now()
        
        update_sql = """
        UPDATE pokemon_cards 
        SET last_price = %s, last_checked = %s, updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
        """
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(update_sql, (price, timestamp, card_id))
            self.connection.commit()
            
            if cursor.rowcount > 0:
                print(f"✅ Updated card ID {card_id} with price ${price:.2f}")
                return True
            else:
                print(f"⚠️ No card found with ID {card_id}")
                return False
        except psycopg2.Error as e:
            self.connection.rollback()
            print(f"❌ Error updating card: {e}")
            raise
    
    def get_card_by_name(self, card_name: str) -> Optional[CardRecord]:
        """Get card by name"""
        select_sql = "SELECT * FROM pokemon_cards WHERE card_name = %s"
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(select_sql, (card_name,))
            row = cursor.fetchone()
            
            if row:
                return CardRecord(
                    id=row[0],
                    card_name=row[1],
                    image_path=row[2],
                    last_price=float(row[3]) if row[3] else None,
                    last_checked=row[4]
                )
            return None
        except psycopg2.Error as e:
            print(f"❌ Error getting card by name: {e}")
            raise
    
    def get_card_by_id(self, card_id: int) -> Optional[CardRecord]:
        """Get card by ID"""
        select_sql = "SELECT * FROM pokemon_cards WHERE id = %s"
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(select_sql, (card_id,))
            row = cursor.fetchone()
            
            if row:
                return CardRecord(
                    id=row[0],
                    card_name=row[1],
                    image_path=row[2],
                    last_price=float(row[3]) if row[3] else None,
                    last_checked=row[4]
                )
            return None
        except psycopg2.Error as e:
            print(f"❌ Error getting card by ID: {e}")
            raise
    
    def get_all_cards(self) -> List[CardRecord]:
        """Get all cards"""
        select_sql = "SELECT * FROM pokemon_cards ORDER BY card_name"
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(select_sql)
            rows = cursor.fetchall()
            
            cards = []
            for row in rows:
                cards.append(CardRecord(
                    id=row[0],
                    card_name=row[1],
                    image_path=row[2],
                    last_price=float(row[3]) if row[3] else None,
                    last_checked=row[4]
                ))
            return cards
        except psycopg2.Error as e:
            print(f"❌ Error getting all cards: {e}")
            raise
    
    def delete_card(self, card_id: int) -> bool:
        """Delete a card by ID"""
        delete_sql = "DELETE FROM pokemon_cards WHERE id = %s"
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(delete_sql, (card_id,))
            self.connection.commit()
            
            if cursor.rowcount > 0:
                print(f"✅ Deleted card with ID {card_id}")
                return True
            else:
                print(f"⚠️ No card found with ID {card_id}")
                return False
        except psycopg2.Error as e:
            self.connection.rollback()
            print(f"❌ Error deleting card: {e}")
            raise


class DatabaseFactory:
    """Factory class to create database instances"""
    
    @staticmethod
    def create_sqlite_db(db_path: str = "pokemon_cards.db") -> SQLiteDatabase:
        """Create SQLite database instance"""
        return SQLiteDatabase(db_path)
    
    @staticmethod
    def create_postgresql_db(host: str, database: str, user: str, password: str, port: int = 5432) -> PostgreSQLDatabase:
        """Create PostgreSQL database instance"""
        return PostgreSQLDatabase(host, database, user, password, port)
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> DatabaseInterface:
        """Create database instance from configuration dictionary"""
        db_type = config.get('type', 'postgresql').lower()
        
        if db_type == 'sqlite':
            return DatabaseFactory.create_sqlite_db(config.get('path', 'pokemon_cards.db'))
        elif db_type == 'postgresql':
            return DatabaseFactory.create_postgresql_db(
                host=config['host'],
                database=config['database'],
                user=config['user'],
                password=config['password'],
                port=config.get('port', 5432)
            )
        else:
            raise ValueError(f"Unsupported database type: {db_type}")


# Example usage and testing
if __name__ == "__main__":
    # SQLite example
    print("=== SQLite Database Example ===")
    
    # Create SQLite database
    db = DatabaseFactory.create_sqlite_db("test_pokemon_cards.db")
    
    # Create a test card
    test_card = CardRecord(
        card_name="Base Set/Charizard",
        image_path="/path/to/charizard.jpg",
        last_price=299.99,
        last_checked=datetime.now()
    )
    
    # Insert card
    card_id = db.insert_card(test_card)
    
    # Get card by name
    found_card = db.get_card_by_name("Base Set/Charizard")
    if found_card:
        print(f"Found card: {found_card.card_name} - ${found_card.last_price}")
    
    # Update price
    db.update_card_price(card_id, 350.00)
    
    # Get all cards
    all_cards = db.get_all_cards()
    print(f"Total cards in database: {len(all_cards)}")
    
    # Clean up
    db.disconnect()
    
    # Clean up test file
    if os.path.exists("test_pokemon_cards.db"):
        os.remove("test_pokemon_cards.db")
    
    print("\n=== Configuration Example ===")
    
    # Configuration-based approach
    sqlite_config = {
        'type': 'sqlite',
        'path': 'pokemon_cards.db'
    }
    
    postgresql_config = {
        'type': 'postgresql',
        'host': '35.95.111.21',
        'database': 'pokemoncards',
        'user': 'pokemon',
        'password': 'pokemon',
        'port': 5432
    }
    
    # You can easily switch between databases by changing the config
    # db = DatabaseFactory.create_from_config(sqlite_config)
    # db = DatabaseFactory.create_from_config(postgresql_config)
    
    print("✅ Database layer ready for use!")