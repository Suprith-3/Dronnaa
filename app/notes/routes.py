import os
import uuid
from flask import render_template, request, redirect, url_for, flash, current_app
from flask_login import current_user, login_required
from . import notes_bp
from ..models import Note, db
from werkzeug.utils import secure_filename
from modules.supabase_helper import upload_file_to_supabase

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@notes_bp.route('/')
def index():
    notes = Note.query.order_by(Note.date_uploaded.desc()).all()
    # Unique subjects for filtering
    subjects = db.session.query(Note.subject).distinct().all()
    subjects = [s[0] for s in subjects]
    return render_template('notes/index.html', notes=notes, subjects=subjects)

@notes_bp.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            
            final_path = unique_filename
            # Try Supabase first
            if current_app.config.get('SUPABASE_URL'):
                supa_url = upload_file_to_supabase('dronacharya', file, f"notes/{unique_filename}")
                if supa_url:
                    final_path = supa_url
                else:
                    file.save(os.path.join(current_app.config['NOTES_FOLDER'], unique_filename))
            else:
                file.save(os.path.join(current_app.config['NOTES_FOLDER'], unique_filename))

            
            new_note = Note(
                title=request.form.get('title'),
                subject=request.form.get('subject'),
                description=request.form.get('description'),
                file_path=final_path,
                uploader_name=current_user.first_name if current_user.is_authenticated else request.form.get('uploader_name', 'Anonymous')
            )
            
            db.session.add(new_note)
            db.session.commit()
            flash('Note uploaded successfully!', 'success')
            return redirect(url_for('notes.index'))
        else:
            flash('Allowed file types are: png, jpg, jpeg, gif, webp, pdf', 'danger')
            
    return render_template('notes/upload.html')
