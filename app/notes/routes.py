import os
import uuid
from flask import render_template, request, redirect, url_for, flash, current_app
from flask_login import current_user, login_required
from . import notes_bp
from ..models import Note, UserCoupon, db
from modules.text_generation import classify_document_type
from werkzeug.utils import secure_filename
from modules.supabase_helper import upload_file_to_supabase
import random

COUPON_LINKS = [
    "https://bitli.in/XArgSgZ", "https://ajiio.in/ahKYk2s", "https://ajiio.in/wOMX1Hp",
    "https://ajiio.in/YjgmyRN", "https://fktr.in/rs6zW0E", "https://fktr.in/4cuBCro",
    "https://bitli.in/nO43Rc9", "https://bitli.in/UAc8MyM", "https://bitli.in/EObBZT3",
    "https://bitli.in/0MypuLb", "https://bitli.in/o305qGK", "https://bitli.in/HaEa1En",
    "https://myntr.it/2cqqwyK", "https://myntr.it/9Ipzf4E", "https://myntr.it/0Dli4Pq",
    "https://myntr.it/J3978Uw", "https://myntr.it/8baazC4", "https://myntr.it/ffYgvgS",
    "https://myntr.it/v55GoJS", "https://myntr.it/ZzXI0dF", "https://myntr.it/w1wIJzI",
    "https://myntr.it/dctedyT", "https://myntr.it/mSgCTwr", "https://myntr.it/jzXVkGn",
    "https://myntr.it/sI1t8kk", "https://myntr.it/II4Rj7V", "https://myntr.it/qR78QsQ",
    "https://myntr.it/e8xEk7Y"
]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@notes_bp.route('/')
def index():
    notes = Note.query.order_by(Note.date_uploaded.desc()).all()
    handwritten_notes = [n for n in notes if n.is_handwritten]
    printed_notes = [n for n in notes if not n.is_handwritten]
    
    # Unique subjects for filtering
    subjects = db.session.query(Note.subject).distinct().all()
    subjects = [s[0] for s in subjects]
    return render_template('notes/index.html', notes=notes, handwritten_notes=handwritten_notes, printed_notes=printed_notes, subjects=subjects)

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
                    # Fallback to local
                    file.save(os.path.join(current_app.config['NOTES_FOLDER'], unique_filename))
                    flash('Note saved locally. Warning: Permanence not guaranteed (Supabase storage error).', 'warning')
            else:
                # Local only
                file.save(os.path.join(current_app.config['NOTES_FOLDER'], unique_filename))
                flash('Note saved locally. Configure Supabase for permanent storage.', 'info')


            
            # Perform classification
            temp_path = os.path.join(current_app.config['NOTES_FOLDER'], f"temp_{unique_filename}")
            file.seek(0)
            file.save(temp_path)
            
            # User proposed type
            user_proposed_type = request.form.get('note_type', 'handwritten')
            
            # AI Check
            try:
                is_detected_handwritten = classify_document_type(temp_path)
            except:
                is_detected_handwritten = (user_proposed_type == 'handwritten')
            
            # Finalize note model
            new_note = Note(
                title=request.form.get('title'),
                subject=request.form.get('subject'),
                description=request.form.get('description'),
                file_path=final_path,
                uploader_name=current_user.first_name if current_user.is_authenticated else request.form.get('uploader_name', 'Anonymous'),
                user_id=current_user.id if current_user.is_authenticated else None,
                is_handwritten=is_detected_handwritten
            )
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            db.session.add(new_note)
            
            # Award Coupon
            coupon_to_award = None
            if current_user.is_authenticated:
                # Find coupons user hasn't received yet
                user_coupons = {c.coupon_url for c in current_user.coupons}
                available_coupons = [url for url in COUPON_LINKS if url not in user_coupons]
                
                if not available_coupons:
                    # If all earned, pick a random one to restart cycle or just random
                    coupon_to_award = random.choice(COUPON_LINKS)
                else:
                    coupon_to_award = random.choice(available_coupons)
                
                new_coupon = UserCoupon(user_id=current_user.id, coupon_url=coupon_to_award)
                db.session.add(new_coupon)
            
            db.session.commit()
            
            if coupon_to_award:
                flash(f'Note uploaded! You earned a reward: {coupon_to_award}', 'success')
                # We'll use a session flag to trigger the modal on the next page
                from flask import session
                session['new_coupon_url'] = coupon_to_award
            else:
                flash('Note uploaded successfully!', 'success')
                
            return redirect(url_for('notes.index'))
        else:
            flash('Allowed file types are: png, jpg, jpeg, gif, webp, pdf', 'danger')
            
    return render_template('notes/upload.html')

@notes_bp.route('/delete/<int:note_id>', methods=['POST'])
@login_required
def delete_note(note_id):
    note = Note.query.get_or_404(note_id)
    
    # Permission check: Only owner or admin (if you have admins) can delete
    if note.user_id != current_user.id:
        flash('You do not have permission to delete this note.', 'danger')
        return redirect(url_for('notes.index'))
    
    try:
        # 1. Delete actual file if it exists
        if note.file_path.startswith('http'):
            # It's a Supabase URL, try to extract path and delete
            # Supabase URLs look like: https://.../storage/v1/object/public/bucket/notes/unique_filename
            from modules.supabase_helper import delete_file_from_supabase
            path_part = note.file_path.split('/notes/')[-1]
            delete_file_from_supabase('dronacharya', f"notes/{path_part}")
        else:
            # Local file
            local_path = os.path.join(current_app.config['NOTES_FOLDER'], note.file_path)
            if os.path.exists(local_path):
                os.remove(local_path)
        
        # 2. Delete from DB
        db.session.delete(note)
        db.session.commit()
        flash('Note deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting note: {str(e)}', 'danger')
        
    return redirect(url_for('notes.index'))

