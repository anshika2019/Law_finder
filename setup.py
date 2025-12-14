"""
Setup script for IPC Section Query System
Run this once to set up the database
"""
import os
from chroma_conn import create_database, get_database

def main():
    print("="*70)
    print("IPC Section Query System - Setup")
    print("="*70)
    print("\nChecking database...")
    
    # Check if database exists
    db_path = os.path.join(os.path.dirname(__file__), 'chroma_langchain_db', 'chroma.sqlite3')
    
    if os.path.exists(db_path):
        print("✅ Database already exists!")
        print("Verifying database...")
        try:
            db = get_database()
            print("✅ Database is valid and ready to use!")
        except Exception as e:
            print(f"⚠️  Database exists but has errors: {e}")
            print("Recreating database...")
            db = create_database()
            print("✅ Database recreated successfully!")
    else:
        print("⚠️  Database not found. Creating new database...")
        print("This may take a few minutes...")
        db = create_database()
        print("✅ Database created successfully!")
    
    print("\n" + "="*70)
    print("Setup complete! You can now run:")
    print("  streamlit run app.py")
    print("="*70)

if __name__ == "__main__":
    main()

