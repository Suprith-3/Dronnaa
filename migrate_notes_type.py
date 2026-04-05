from app import db, create_app
from sqlalchemy import text

app = create_app()
with app.app_context():
    print("Checking for is_handwritten column in Note table...")
    try:
        # Check if column exists
        with db.engine.connect() as conn:
            conn.execute(text("SELECT is_handwritten FROM note LIMIT 1"))
        print("Column 'is_handwritten' already exists.")
    except Exception:
        print("Column 'is_handwritten' does not exist. Adding it...")
        try:
            with db.engine.connect() as conn:
                # PostgreSQL syntax
                conn.execute(text("ALTER TABLE note ADD COLUMN is_handwritten BOOLEAN DEFAULT FALSE"))
                conn.commit()
            print("Successfully added 'is_handwritten' column.")
        except Exception as e:
            print(f"Error adding column: {e}")
            # Try SQLite or if commit is handled differently
            try:
                with db.engine.connect() as conn:
                    conn.execute(text("ALTER TABLE note ADD COLUMN is_handwritten BOOLEAN DEFAULT 0"))
                print("Successfully added 'is_handwritten' column (no-commit variant).")
            except Exception as e2:
                print(f"Final error: {e2}")

    print("Migration complete.")
