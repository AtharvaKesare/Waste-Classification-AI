import sqlite3
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
DB_FILE = "waste_data.db"

def init_db():
    """Initializes the database with the required schema."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL NOT NULL,
                feedback INTEGER DEFAULT NULL, 
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # feedback: 1 for Correct, 0 for Incorrect, NULL for no feedback
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

def save_prediction(filename, category, confidence):
    """Saves a prediction to the database and returns the inserted ID."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (filename, category, confidence)
            VALUES (?, ?, ?)
        ''', (filename, category, confidence))
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return prediction_id
    except Exception as e:
        logger.error(f"Failed to save prediction: {e}")
        return None

def update_feedback(prediction_id, is_correct):
    """Updates the feedback for a specific prediction."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        feedback_value = 1 if is_correct else 0
        cursor.execute('''
            UPDATE predictions 
            SET feedback = ? 
            WHERE id = ?
        ''', (feedback_value, prediction_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to update feedback: {e}")
        return False

def get_analytics():
    """Retrieves analytics data for the dashboard."""
    stats = {
        'total': 0,
        'organic': 0,
        'recycle': 0,
        'non_waste': 0,
        'feedback_received': 0,
        'accuracy': 0.0
    }
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Get total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        stats['total'] = cursor.fetchone()[0]
        
        # Get category breakdown
        cursor.execute("SELECT category, COUNT(*) FROM predictions GROUP BY category")
        for row in cursor.fetchall():
            if row[0] == 'Organic': stats['organic'] = row[1]
            elif row[0] == 'Recycle': stats['recycle'] = row[1]
            elif row[0] == 'Non-Waste': stats['non_waste'] = row[1]
            
        # Get accuracy (correct feedbacks / total feedbacks)
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE feedback IS NOT NULL")
        stats['feedback_received'] = cursor.fetchone()[0]
        
        if stats['feedback_received'] > 0:
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE feedback = 1")
            correct = cursor.fetchone()[0]
            stats['accuracy'] = round((correct / stats['feedback_received']) * 100, 1)
            
        conn.close()
    except Exception as e:
        logger.error(f"Failed to fetch analytics: {e}")
        
    return stats
