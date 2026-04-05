import os, json, uuid, random
from concurrent.futures import ThreadPoolExecutor
from flask import render_template, redirect, url_for, request, flash, jsonify, session, current_app, send_file
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from . import tutor_bp
from ..models import User, Course, Video, Quiz, QuizHistory, UserCoupon, Roadmap, RoadmapTask, db
from modules.text_generation import generate_gemini_quiz, generate_gemini_chat, generate_gemini_notes, generate_gemini_vision, generate_gemini_courses, generate_specific_course, generate_roadmap_tasks
from modules.text_to_speech import generate_tts
from modules.video_search import search_videos
from modules.document_generator import create_docx
from modules.supabase_helper import upload_file_to_supabase
from modules.summary_helper import generate_ai_summary
from datetime import datetime

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
            flash('Login successful! You have 20 new scholarship opportunities waiting for you.', 'info')
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
        flash('Welcome to Dronacharya Hub! We have found 20 scholarships to help your educational journey.', 'info')
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
    # Only show the permanent 23 courses in the dashboard
    courses = Course.query.filter_by(is_permanent=True).options(joinedload(Course.videos)).all()
    
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
    query = request.args.get('q')
    if query:
        # Search ALL courses (including previously searched ones)
        courses = Course.query.filter(Course.title.ilike(f'%{query}%')).all()
        
        # If none found, generate one specifically for this topic
        if not courses:
            try:
                item = generate_specific_course(query)
                # Check if this title was already generated/saved by another user but didn't show in ILIKE search exactly
                existing = Course.query.filter_by(title=item.get('title')).first()
                if existing:
                    courses = [existing]
                else:
                    c = Course(
                        title=item.get('title'),
                        description=item.get('description'),
                        level=item.get('level', 'Beginner')
                    )
                    db.session.add(c)
                    db.session.flush()
                    
                    # Add a video
                    yt_res = search_videos(f"{c.title} educational lecture", 1)
                    v_url = yt_res[0]['video_url'] if yt_res else "https://www.youtube.com/watch?v=rfscVS0vtbw"
                    v_title = yt_res[0]['title'] if yt_res else f"Introduction to {c.title}"
                    
                    v = Video(course_id=c.id, title=v_title, video_url=v_url)
                    db.session.add(v)
                    db.session.commit()
                    courses = [c]
            except Exception as e:
                print(f"Error generating specific course: {e}")
                courses = []
    else:
        # ONLY show the permanent 23 courses by default
        courses = Course.query.filter_by(is_permanent=True).all()
        
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

@tutor_bp.route('/robite')
@login_required
def robite_page():
    # Dedicated "Friend Chat" page
    return render_template('tutor/robite.html')

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
            supa_url = upload_file_to_supabase(current_app.config.get('SUPABASE_BUCKET', 'dronacharya'), f, f"audio/{filename}")
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
            supa_url = upload_file_to_supabase(current_app.config.get('SUPABASE_BUCKET', 'dronacharya'), f, f"notes/{filename}")
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
                    supa_url = upload_file_to_supabase(current_app.config.get('SUPABASE_BUCKET', 'dronacharya'), file, f"avatars/{filename}")
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

@tutor_bp.route('/api/more-courses', methods=['POST'])
@login_required
def api_more_courses():
    try:
        new_data = generate_gemini_courses()
        if not new_data:
            return jsonify({'success': True, 'courses': []})

        # Step 1: Filter out duplicates in ONE query
        new_titles = [item.get('title') for item in new_data]
        existing_titles = {c.title for c in Course.query.filter(Course.title.in_(new_titles)).all()}
        unique_new_data = [item for item in new_data if item.get('title') not in existing_titles]
        
        # FINAL SAFETY NET: If everything was a duplicate, just force uniqueness by renaming!
        if not unique_new_data:
            for item in new_data:
                # Add a unique identifier tag to make it unique in the DB
                item['title'] = f"{item['title']} - Edition {random.randint(2, 999)}"
            unique_new_data = new_data
        
        # Step 2: Parallel Video Search (Network Intensive)
        def fetch_video_metadata(item):
            title = item.get('title')
            yt_res = search_videos(f"{title} educational lecture", 1)
            return {
                'item': item,
                'video': yt_res[0] if yt_res else None
            }

        with ThreadPoolExecutor(max_workers=5) as executor:
            meta_results = list(executor.map(fetch_video_metadata, unique_new_data))
        
        # Step 3: Batch Save to DB
        added_courses = []
        for res in meta_results:
            item = res['item']
            video_data = res['video']
            
            c = Course(
                title=item.get('title'),
                description=item.get('description', ''),
                level=item.get('level', 'Beginner'),
                is_permanent=True
            )
            db.session.add(c)
            db.session.flush() # Get ID
            
            v_url = video_data['video_url'] if video_data else "https://www.youtube.com/watch?v=rfscVS0vtbw"
            v_title = video_data['title'] if video_data else f"Introduction to {c.title}"
            
            v = Video(course_id=c.id, title=v_title, video_url=v_url)
            db.session.add(v)
            
            added_courses.append({
                'id': c.id,
                'title': c.title,
                'description': c.description,
                'level': c.level,
                'video_title': v_title,
                'video_url': v_url
            })
            
        db.session.commit()
        return jsonify({'success': True, 'courses': added_courses})
    except Exception as e:
        db.session.rollback()
        print(f"More Courses Error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@tutor_bp.route('/rewards')
@login_required
def rewards():
    # Mapping URLs to brands for better display
    brand_map = {
        'bitli.in': 'Special Reward',
        'ajiio.in': 'AJIO',
        'fktr.in': 'Flipkart',
        'myntr.it': 'Myntra'
    }
    
    coupons = UserCoupon.query.filter_by(user_id=current_user.id).order_by(UserCoupon.date_earned.desc()).all()
    
    formatted_coupons = []
    for c in coupons:
        brand = 'Mystery Coupon'
        for key, name in brand_map.items():
            if key in c.coupon_url:
                brand = name
                break
        formatted_coupons.append({
            'url': c.coupon_url,
            'date': c.date_earned.strftime('%d %b %Y'),
            'brand': brand
        })
        
    return render_template('tutor/rewards.html', coupons=formatted_coupons)

@tutor_bp.route('/summary')
@login_required
def summary_page():
    return render_template('tutor/summary.html')

@tutor_bp.route('/api/generate-summary', methods=['POST'])
@login_required
def api_generate_summary():
    data = request.get_json()
    url = data.get('url')
    language = data.get('language', 'English')
    
    if not url:
        return jsonify({'success': False, 'message': 'URL is required'})
        
    summary = generate_ai_summary(url, language)
    return jsonify({'success': True, 'summary': summary})

@tutor_bp.route('/live-session')
@tutor_bp.route('/live-session/<session_id>')
@login_required
def live_session(session_id=None):
    return render_template('tutor/live_session.html', session_id=session_id)

@tutor_bp.route('/live-chat', methods=['POST'])
@login_required
def live_chat():
    user_msg = request.json.get('message', '')
    if not user_msg:
        return jsonify({'error': 'No message provided'}), 400
    
    # Custom prompt for "friend" persona
    friendly_prompt = f"""
    You are ROBITE, a high-tech robotic friend. 
    A user is talking to you: "{user_msg}"

    RULES FOR MAX SPEED (Target <2000ms):
    1. EXTREMELY BRIEF: Your reply MUST be 15-20 words max. Be snappy and concise. 
    2. HUMAN-LIKE: Still be friendly and use emojis. Talk like a real friend.
    3. LANGUAGE: English only.
    4. GRAMMAR: Correct user grammar mistakes ONLY if they are major, but keep your own response super short!
    5. ENGAGING: One short sentence + one quick question.
    """
    
    try:
        from modules.text_generation import generate_with_fallback
        # Use the robust fallback generator for the persona response
        response_text = generate_with_fallback(friendly_prompt)
        
        # Generate TTS for voice response
        import uuid
        filename = f"robite_{uuid.uuid4().hex[:6]}.mp3"
        filepath = os.path.join(current_app.config['AUDIO_FOLDER'], filename)
        generate_tts(response_text, filepath)
        audio_url = url_for('static', filename='audio/' + filename)

        return jsonify({'response': response_text, 'audio_url': audio_url})
    except Exception as e:
        print(f"Robite Fallback Error: {e}")
        # Final safety net with hardcoded friendly responses
        safe_responses = [
            "Hey! My brain took a tiny nap for a second. Can you tell me that again? 😊",
            "Oops, I got a bit distracted by some code! What were we talking about? 🧡",
            "My AI senses are tingling, but I missed that last part. Say it again? ✨",
            "Just resetting my circuits! What was that, friend? 🤖"
        ]
        response_text = random.choice(safe_responses)
        
        # Ensure we still have a TTS file for the fallback response
        try:
            import uuid
            filename = f"robite_fb_{uuid.uuid4().hex[:6]}.mp3"
            filepath = os.path.join(current_app.config['AUDIO_FOLDER'], filename)
            generate_tts(response_text, filepath)
            audio_url = url_for('static', filename='audio/' + filename)
            return jsonify({'response': response_text, 'audio_url': audio_url})
        except:
            return jsonify({'response': response_text})

@tutor_bp.route('/api/live-translate', methods=['POST'])
@login_required
def api_live_translate():
    data = request.get_json()
    text = data.get('text', '')
    language = data.get('language', 'English')
    
    if not text:
        return jsonify({'success': False, 'translation': ''})
        
    try:
        from modules.text_generation import generate_with_fallback
        prompt = f"Translate the following text into {language}. Return ONLY the translated text, no quotation marks, no preamble:\n\n{text}"
        translated_text = generate_with_fallback(prompt)
        return jsonify({'success': True, 'translation': translated_text.strip()})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e), 'translation': f'[Error: {str(e)[:20]}]'})

@tutor_bp.route('/api/live-summary', methods=['POST'])
@login_required
def api_live_summary():
    data = request.get_json()
    transcript = data.get('transcript', '')
    language = data.get('language', 'English')
    
    if not transcript or len(transcript) < 10:
        return jsonify({'success': False, 'message': 'Transcript too short.'})
        
    try:
        from modules.text_generation import generate_with_fallback
        prompt = f"Provide a comprehensive structured summary with bullet points of the following live class transcript in {language}:\n\n{transcript[:10000]}"
        summary_text = generate_with_fallback(prompt)
        return jsonify({'success': True, 'summary': summary_text})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@tutor_bp.route('/api/download-live-doc', methods=['POST'])
@login_required
def download_live_doc():
    transcript = request.form.get('transcript', '')
    summary = request.form.get('summary', '')
    
    content = f"--- DRONACHARYA HUB LIVE SESSION ---\n\n## SUMMARY ##\n{summary}\n\n## FULL TRANSCRIPT ##\n{transcript}"
    
    import io, uuid
    filename = f"Live_Session_{uuid.uuid4().hex[:6]}.txt"
    filepath = os.path.join(current_app.config.get('TEMP_FOLDER', '/tmp'), filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
    return send_file(filepath, as_attachment=True, download_name=filename)

# ── RODE MAP (AI Roadmap) ──────────────────────────────────────────

@tutor_bp.route('/roadmap')
@login_required
def roadmap():
    roadmaps = Roadmap.query.filter_by(user_id=current_user.id).order_by(Roadmap.created_at.desc()).all()
    return render_template('tutor/roadmap.html', roadmaps=roadmaps)

@tutor_bp.route('/api/generate-roadmap', methods=['POST'])
@login_required
def api_generate_roadmap():
    subject = request.json.get('subject')
    months = request.json.get('months', 1)
    
    if not subject: return jsonify({'success': False, 'message': 'Missing subject'}), 400
    
    try:
        tasks_data = generate_roadmap_tasks(subject, months)
        if not tasks_data: return jsonify({'success': False, 'message': 'AI failed to generate tasks.'}), 500
        
        new_roadmap = Roadmap(user_id=current_user.id, subject=subject, duration=months)
        db.session.add(new_roadmap)
        db.session.flush() 
        
        for t in tasks_data:
            task = RoadmapTask(
                roadmap_id=new_roadmap.id,
                day_number=t.get('day', 1),
                topic=t.get('topic', 'Learning Session'),
                description=t.get('description', '')
            )
            db.session.add(task)
            
        db.session.commit()
        return jsonify({'success': True, 'roadmap_id': new_roadmap.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

@tutor_bp.route('/api/toggle-task/<int:task_id>', methods=['POST'])
@login_required
def toggle_task(task_id):
    task = RoadmapTask.query.get_or_404(task_id)
    if task.roadmap_parent.user_id != current_user.id:
        return jsonify({'success': False}), 403
    
    task.is_completed = not task.is_completed
    task.completed_at = datetime.utcnow() if task.is_completed else None
    db.session.commit()
    return jsonify({'success': True, 'is_completed': task.is_completed})
