import os
from flask import render_template, request, redirect, url_for, session, flash, current_app
from functools import wraps
from . import admin_bp
from ..models import User, Note, Book
from .. import db

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('Please log in to access the admin panel.', 'warning')
            return redirect(url_for('admin.login'))
        return f(*args, **kwargs)
    return decorated_function

@admin_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '')
        password = request.form.get('password', '')
        if email.strip() == 'supreethm763@gmail.com' and password == 'suprith@974244':
            session['admin_logged_in'] = True
            flash('Successfully logged in as Admin', 'success')
            return redirect(url_for('admin.dashboard'))
        flash('Invalid admin credentials', 'danger')
    return render_template('admin/login.html')

@admin_bp.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('admin.login'))

@admin_bp.route('/dashboard')
@admin_required
def dashboard():
    users = User.query.all()
    notes = Note.query.all()
    books = Book.query.all()
    
    # Calculate marketplace data
    total_revenue = sum([book.price for book in books if getattr(book, 'price', 0)])
    users_with_research_access = User.query.filter_by(has_research_access=True).all()
    
    return render_template(
        'admin/dashboard.html', 
        users=users, 
        notes=notes, 
        books=books,
        total_revenue=total_revenue,
        users_with_research_access=users_with_research_access
    )

@admin_bp.route('/delete_user/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    # Manual cascade delete for models that might block deletion
    try:
        if hasattr(user, 'books'):
            for book in user.books:
                db.session.delete(book)
        if hasattr(user, 'notes'):
            for note in user.notes:
                db.session.delete(note)
        
        db.session.delete(user)
        db.session.commit()
        flash('User deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting user: {str(e)}', 'danger')
    return redirect(url_for('admin.dashboard'))

@admin_bp.route('/delete_book/<int:book_id>', methods=['POST'])
@admin_required
def delete_book(book_id):
    book = Book.query.get_or_404(book_id)
    try:
        db.session.delete(book)
        db.session.commit()
        flash('Book deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting book: {str(e)}', 'danger')
    return redirect(url_for('admin.dashboard'))

@admin_bp.route('/delete_note/<int:note_id>', methods=['POST'])
@admin_required
def delete_note(note_id):
    note = Note.query.get_or_404(note_id)
    try:
        # Also remove file if needed
        if note.file_path and os.path.exists(os.path.join(current_app.config['NOTES_FOLDER'], os.path.basename(note.file_path))):
            try:
                os.remove(os.path.join(current_app.config['NOTES_FOLDER'], os.path.basename(note.file_path)))
            except:
                pass
        db.session.delete(note)
        db.session.commit()
        flash('Note deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting note: {str(e)}', 'danger')
    return redirect(url_for('admin.dashboard'))
