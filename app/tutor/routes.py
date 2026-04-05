import os, json, uuid
from flask import render_template, redirect, url_for, request, flash, jsonify, session, current_app, send_file
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from . import tutor_bp
from ..models import User, Course, Video, Quiz, QuizHistory, db
from modules.text_generation import generate_gemini_quiz, generate_gemini_chat, generate_gemini_notes, generate_gemini_vision
from modules.text_to_speech import generate_tts
from modules.video_search import search_videos
from modules.document_generator import create_docx
from modules.supabase_helper import upload_file_to_supabase

@tutor_bp.route('/')
def welcome():
    if current_user.is_authenticated:
        return redirect(url_for('tutor.dashboard'))
    return render_template('tutor/welcome.html')

@tutor_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form.get('email')).first()
        if user and user.check_password(request.form.get('password')):
            login_user(user)
            return redirect(url_for('tutor.dashboard'))
        flash('Invalid email or password', 'danger')
    return render_template('tutor/login.html')

@tutor_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if User.query.filter_by(email=request.form.get('email')).first():
            flash('Email already exists', 'danger')
            return redirect(url_for('tutor.register'))
        new_user = User(
            first_name=request.form.get('first_name'),
            last_name=request.form.get('last_name'),
            email=request.form.get('email'),
            dob=request.form.get('dob'),
            password=request.form.get('password')
        )
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('tutor.dashboard'))
    return render_template('tutor/register.html')

@tutor_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('tutor.welcome'))

from sqlalchemy.orm import joinedload

@tutor_bp.route('/dashboard')
@login_required
def dashboard():
    # Fetch all courses with their videos eager-loaded (to avoid N+1 when counting total videos)
    courses = Course.query.options(joinedload(Course.videos)).all()
    
    # Pre-fetch all completed video IDs for the current user in a single query
    completed_video_ids = {v.id for v in current_user.completed_videos}
    
    progress_data = []
    for c in courses:
        total = len(c.videos)
        # Count completed videos for this specific course from the pre-fetched set
        done_count = sum(1 for v in c.videos if v.id in completed_video_ids)
        
        progress = int((done_count / total) * 100) if total > 0 else 0
        progress_data.append({'course': c, 'progress': progress})
        
    return render_template('tutor/dashboard.html', progress_data=progress_data)

@tutor_bp.route('/courses')
def courses():
    courses = Course.query.all()
    return render_template('tutor/courses.html', courses=courses)

@tutor_bp.route('/courses/<int:id>')
def course_detail(id):
    course = Course.query.get_or_404(id)
    return render_template('tutor/course_detail.html', course=course)

@tutor_bp.route('/video/<int:id>')
@login_required
def video_detail(id):
    video = Video.query.get_or_404(id)
    return render_template('tutor/video_detail.html', video=video)

@tutor_bp.route('/video/<int:id>/complete', methods=['POST'])
@login_required
def complete_video(id):
    video = Video.query.get_or_404(id)
    if video not in current_user.completed_videos:
        current_user.completed_videos.append(video)
        db.session.commit()
    return jsonify({'success': True})

@tutor_bp.route('/quiz')
@login_required
def quiz_page():
    return render_template('tutor/quiz.html')

@tutor_bp.route('/api/generate-quiz', methods=['POST'])
@login_required
def api_generate_quiz():
    topic = request.json.get('topic')
    questions = generate_gemini_quiz(topic)
    return jsonify({'success': True, 'questions': questions})

@tutor_bp.route('/api/save-quiz', methods=['POST'])
@login_required
def api_save_quiz():
    data = request.json
    history = QuizHistory(
        user_id=current_user.id,
        topic=data.get('topic'),
        score=data.get('score'),
        total_questions=data.get('total'),
        percentage=data.get('percentage'),
        details=json.dumps(data.get('details'))
    )
    db.session.add(history)
    db.session.commit()
    return jsonify({'success': True})

@tutor_bp.route('/quiz/history')
@login_required
def quiz_history():
    history = QuizHistory.query.filter_by(user_id=current_user.id).order_by(QuizHistory.date_taken.desc()).all()
    return render_template('tutor/quiz_history.html', history=history)

@tutor_bp.route('/chat')
@login_required
def chat_page():
    session['chat_history'] = []
    return render_template('tutor/chat.html')

@tutor_bp.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    message = request.json.get('message')
    history = session.get('chat_history', [])
    reply, new_history = generate_gemini_chat(message, history)
    session['chat_history'] = new_history
    return jsonify({'reply': reply})

@tutor_bp.route('/get-answer', methods=['POST'])
@login_required
def get_answer():
    question = request.form.get('question')
    # Placeholder for a more complex Q&A if needed, for now use Gemini Chat logic but simple
    reply, _ = generate_gemini_chat(f"Briefly answer: {question}", [])
    
    # Generate TTS
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(current_app.config['AUDIO_FOLDER'], filename)
    generate_tts(reply, filepath)
    
    # Supabase Upload (Optional Persistence)
    audio_url = url_for('static', filename='audio/'+filename)
    if current_app.config.get('SUPABASE_URL'):
        with open(filepath, 'rb') as f:
            supa_url = upload_file_to_supabase('dronacharya', f, f"audio/{filename}")
            if supa_url: audio_url = supa_url

    # Search videos
    videos = search_videos(question)
    
    return render_template('tutor/lesson.html', question=question, answer=reply, audio_url=audio_url, videos=videos)

@tutor_bp.route('/api/generate-notes', methods=['POST'])
@login_required
def api_generate_notes():
    topic = request.json.get('topic')
    notes_md = generate_gemini_notes(topic)
    filename = f"notes_{uuid.uuid4().hex}.docx"
    filepath = os.path.join(current_app.config['DOCS_FOLDER'], filename)
    create_docx(notes_md, filepath)
    
    download_url = url_for('static', filename='documents/'+filename)
    if current_app.config.get('SUPABASE_URL'):
        with open(filepath, 'rb') as f:
            supa_url = upload_file_to_supabase('dronacharya', f, f"notes/{filename}")
            if supa_url: download_url = supa_url

    return jsonify({'success': True, 'download_url': download_url})

@tutor_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.first_name = request.form.get('first_name')
        current_user.last_name = request.form.get('last_name')
        current_user.dob = request.form.get('dob')
        if 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file and file.filename != '':
                filename = secure_filename(f"avatar_{current_user.id}_{file.filename}")
                # Try Supabase first
                if current_app.config.get('SUPABASE_URL'):
                    supa_url = upload_file_to_supabase('dronacharya', file, f"avatars/{filename}")
                    if supa_url:
                        current_user.profile_pic = supa_url
                    else:
                        # Fallback to local
                        file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
                        current_user.profile_pic = filename
                else:
                    # LOCAL SAVE
                    file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
                    current_user.profile_pic = filename

        
        db.session.commit()
        flash('Profile updated!', 'success')
    return render_template('tutor/profile.html')

@tutor_bp.route('/change-password', methods=['POST'])
@login_required
def change_password():
    new_p = request.form.get('new_password')
    current_user.set_password(new_p)
    db.session.commit()
    flash('Password changed!', 'success')
    return redirect(url_for('tutor.profile'))

@tutor_bp.route('/camera')
@login_required
def camera_page():
    return render_template('tutor/camera.html')

@tutor_bp.route('/api/vision', methods=['POST'])
@login_required
def api_vision():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'Empty file'})
    
    filename = f"capture_{uuid.uuid4().hex}.jpg"
    temp_path = os.path.join(current_app.config['TEMP_FOLDER'], filename)
    file.save(temp_path)
    
    try:
        prompt = request.form.get('prompt', "Solve this educational problem or explain the handwriting.")
        answer = generate_gemini_vision(temp_path, prompt)
        return jsonify({'success': True, 'answer': answer})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
