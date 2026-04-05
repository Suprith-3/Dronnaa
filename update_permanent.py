from app import create_app, db
from app.models import Course

def set_permanent():
    app = create_app()
    with app.app_context():
        # SQLite: Boolean might be 0/1 or True/False depending on how it's handled
        # But SQL Alchemy handles objects so this is better
        courses = Course.query.all()
        for c in courses:
            c.is_permanent = True
        db.session.commit()
        print(f"Updated {len(courses)} courses to be permanent.")

if __name__ == '__main__':
    set_permanent()
