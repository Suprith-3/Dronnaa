import sys
import os
# Add the project root to sys.path
sys.path.append('c:/Users/supre/educaton223/dronacharya-hub')

from app import create_app
from app.models import Book

app = create_app()
with app.app_context():
    books = Book.query.all()
    print(f"Total books: {len(books)}")
    for b in books:
        print(f"ID: {b.id}, Title: {b.title}, Image: {b.cover_image}")
