from app import create_app, db
from sqlalchemy import text

def migrate_db():
    app = create_app()
    with app.app_context():
        print("Starting migration...")
        try:
            with db.engine.connect() as conn:
                # Add the column
                conn.execute(text("ALTER TABLE course ADD COLUMN is_permanent BOOLEAN DEFAULT FALSE"))
                conn.commit()
                print("Step 1: Added 'is_permanent' column.")
                
                # Make original courses permanent
                # Assuming the first 23 courses (or all current ones) should be permanent
                conn.execute(text("UPDATE course SET is_permanent = TRUE"))
                conn.commit()
                print("Step 2: Marked existing courses as permanent.")
                
        except Exception as e:
            print(f"Migration Tip: If the column already exists, you can ignore the error - {e}")
        
        print("Migration task completed.")

if __name__ == '__main__':
    migrate_db()
