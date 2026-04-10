import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'evasioneye.db')
print(f"Connecting to database at {db_path}")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check total records before deleting
    cursor.execute("SELECT COUNT(*) FROM detections")
    count_before = cursor.fetchone()[0]
    print(f"Found {count_before} records to delete.")
    
    # Delete all records from detections table
    cursor.execute("DELETE FROM detections")
    
    # Reset the auto-increment counter
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='detections'")
    
    conn.commit()
    print("Successfully deleted all detection data.")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'conn' in locals():
        conn.close()
