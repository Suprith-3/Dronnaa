from .models import Course, Video, User
from modules.video_search import search_videos

def auto_seed_courses(db):
    print("Database is empty. Seeding initial 23 courses...")
    
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

    for title, desc, level in courses_data:
        c = Course(title=title, description=desc, level=level)
        db.session.add(c)
        db.session.flush()

        # Try to find a real educational video
        yt_res = search_videos(f"{title} educational lecture", 1)
        if yt_res:
            v_url = yt_res[0]['video_url']
            v_title = yt_res[0]['title']
        else:
            v_url = "https://www.youtube.com/watch?v=rfscVS0vtbw"
            v_title = f"Introduction to {title}"

        v = Video(course_id=c.id, title=v_title, video_url=v_url)
        db.session.add(v)
    
    db.session.commit()
    print("Database seeded successfully.")
