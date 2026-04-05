from app import create_app, db
from app.models import User, Course, Video, Quiz, Book
import random

def seed_db():
    app = create_app()
    with app.app_context():
        # Clean start
        db.drop_all()
        db.create_all()
        print("Database structure reconstructed.")

        # 1. Users
        u1 = User(first_name="Suprith", last_name="User", email="suprith@example.com", dob="2000-01-01", password="password123")
        u2 = User(first_name="Arjun", last_name="Kumar", email="arjun@example.com", dob="2000-01-10", password="password123")
        db.session.add_all([u1, u2])
        print("Created users (suprith@example.com / password123).")

        # 2. Courses (Total 23)
        courses_data = [
            ("Python for Archers", "Master logic and precision computation.", "Beginner"),
            ("Data Structures & Precision", "Advanced algorithms for optimization.", "Intermediate"),
            ("Web Engineering Hub", "Build robust backends with Flask.", "Advance"),
            ("Quantum Mechanics Intro", "Exploring the quantum world and wave-particle duality.", "Advance"),
            ("Organic Chemistry: Carbon", "Structures, bonding, and reactions of organic molecules.", "Intermediate"),
            ("The Renaissance: European Art", "The rebirth of art, science, and culture.", "Beginner"),
            ("Calculus I: Limits", "Foundations of calculus and derivatives.", "Beginner"),
            ("Intro to Psychology", "Understanding human behavior and mental processes.", "Beginner"),
            ("Artificial Intelligence 101", "Introduction to machine learning and neural networks.", "Intermediate"),
            ("Global Macroeconomics", "Market analysis and international trade systems.", "Intermediate"),
            ("Digital Marketing Mastery", "SEO, Social media, and conversion optimization.", "Beginner"),
            ("Human Anatomy: Skeletal", "Detailed study of the human skeletal system.", "Intermediate"),
            ("Ancient Philosophy", "From Socrates to Aristotle: Core foundations.", "Beginner"),
            ("Genetics & Heredity", "DNA structures and the laws of inheritance.", "Intermediate"),
            ("Climate Change Science", "Environmental impact and sustainability concepts.", "Beginner"),
            ("Linear Algebra for DS", "Matrix math applied to data science.", "Advance"),
            ("20th Century Literature", "Modern classics and literary movements.", "Intermediate"),
            ("Sociology of Societies", "Interactions and social structures globally.", "Beginner"),
            ("Astrophysics: Origins", "Cosmology and the birth of stars.", "Advance"),
            ("Business Ethics & Law", "Professional values and regulatory systems.", "Intermediate"),
            ("Software Engineering", "Development life cycles and architecture.", "Intermediate"),
            ("Photography: Composition", "Visual storytelling through the lens.", "Beginner"),
            ("Creative Writing Workshop", "Unlocking narrative and poetic expression.", "Beginner")
        ]

        from modules.video_search import search_videos

        # 3. Add Courses and Lessons
        for title, desc, level in courses_data:
            c = Course(title=title, description=desc, level=level, is_permanent=True)
            db.session.add(c)
            db.session.flush()

            # Search for a REALLY relevant video
            yt_res = search_videos(f"{title} educational full lecture", 1)
            if yt_res:
                v_url = yt_res[0]['video_url']
                v_title = yt_res[0]['title']
            else:
                v_url = "https://www.youtube.com/watch?v=rfscVS0vtbw" # Fallback
                v_title = f"Intro to {title}"

            v1 = Video(course_id=c.id, title=v_title, video_url=v_url)
            db.session.add(v1)

        print(f"Seeded {len(courses_data)} Courses with REAL dynamic videos.")

        # 4. Books (Empty for live user data)
        books = []
        db.session.add_all(books)
        db.session.commit()
        print("Stocked book marketplace (Clean slate for users).")
        print("DATABASE SEEDED SUCCESSFULLY.")

if __name__ == '__main__':
    seed_db()
