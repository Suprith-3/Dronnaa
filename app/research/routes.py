import os, uuid
try:
    import razorpay
    rzp = razorpay.Client(auth=(os.getenv('RAZORPAY_KEY_ID', ''), os.getenv('RAZORPAY_KEY_SECRET', '')))
except ImportError:
    razorpay = None
    rzp = None

from flask import render_template, request, jsonify, send_file, current_app, after_this_request, session
from flask_login import current_user, login_required
from . import research_bp
from app import db
from modules.text_generation import generate_gemini_paper, generate_gemini_citation, generate_tumtum_chat
import pypandoc

@research_bp.route('/')
@login_required
def index():
    return render_template('research/index.html', razorpay_key_id=os.getenv('RAZORPAY_KEY_ID', ''))

@research_bp.route('/create-order', methods=['POST'])
@login_required
def create_order():
    if not rzp:
        return jsonify({'success': False, 'message': 'Razorpay is not yet installed on this server. Please run pip install razorpay.'}), 500
    try:
        # Amount in paise: 1 INR = 100 paise
        data = { "amount": 100, "currency": "INR", "receipt": f"research_{current_user.id}_{uuid.uuid4().hex[:8]}" }
        order = rzp.order.create(data=data)
        return jsonify({'success': True, 'order_id': order['id']})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400

@research_bp.route('/verify-payment', methods=['POST'])
@login_required
def verify_payment():
    if not rzp:
        return jsonify({'success': False, 'message': 'Payment Verification Unavailable'}), 500
    try:
        params_dict = {
            'razorpay_order_id': request.json.get('razorpay_order_id'),
            'razorpay_payment_id': request.json.get('razorpay_payment_id'),
            'razorpay_signature': request.json.get('razorpay_signature')
        }
        rzp.utility.verify_payment_signature(params_dict)
        
        current_user.has_research_access = True
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': "Payment Verification Failed"}), 400

@research_bp.route('/generate', methods=['POST'])
@login_required
def generate():
    if not current_user.has_research_access:
        return jsonify({'success': False, 'message': "Payment Required"}), 402

    topic = request.json.get('topic')
    language = request.json.get('language', 'English')
    content = generate_gemini_paper(topic, language)
    if content and not content.startswith("ERROR:"):
        return jsonify({'success': True, 'content': content})
    msg = content.replace("ERROR:", "").strip() if content else "Paper generation failed."
    return jsonify({'success': False, 'message': msg}), 400

@research_bp.route('/generate-survey', methods=['POST'])
@login_required
def generate_survey():
    if not current_user.has_research_access:
        return jsonify({'success': False, 'message': "Payment Required"}), 402

    from modules.text_generation import generate_gemini_survey
    topic = request.json.get('topic')
    language = request.json.get('language', 'English')
    content = generate_gemini_survey(topic, language)
    if content and not content.startswith("ERROR:"):
        return jsonify({'success': True, 'content': content})
    msg = content.replace("ERROR:", "").strip() if content else "Survey generation failed."
    return jsonify({'success': False, 'message': msg}), 400

@research_bp.route('/download', methods=['POST'])
@login_required
def download():
    if not current_user.has_research_access:
        return jsonify({'success': False, 'message': "Payment Required"}), 402

    data = request.json
    content = data.get('content')
    fmt = data.get('format', 'docx').lower()
    
    uid = str(uuid.uuid4())
    temp_md = os.path.join(current_app.config['TEMP_FOLDER'], f'input_{uid}.md')
    ext = 'pdf' if fmt == 'pdf' else 'docx' if fmt == 'docx' else 'txt'
    temp_out = os.path.join(current_app.config['TEMP_FOLDER'], f'output_{uid}.{ext}')
    
    try:
        with open(temp_md, 'w', encoding='utf-8') as f:
            f.write(content)
        
        pypandoc.convert_file(temp_md, fmt if fmt != 'txt' else 'plain', outputfile=temp_out)
        
        @after_this_request
        def cleanup(response):
            try:
                if os.path.exists(temp_md): os.remove(temp_md)
            except: pass
            return response
            
        return send_file(temp_out, as_attachment=True, download_name=f'survey_paper.{ext}')
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@research_bp.route('/generate-citation', methods=['POST'])
@login_required
def generate_citation():
    if not current_user.has_research_access:
        return jsonify({'success': False, 'message': "Payment Required"}), 402

    source = request.json.get('source')
    style = request.json.get('style', 'APA')
    citation = generate_gemini_citation(source, style)
    if citation:
        return jsonify({'success': True, 'citation': citation})
    return jsonify({'success': False, 'message': 'Citation generation failed.'}), 500

@research_bp.route('/chat', methods=['POST'])
@login_required
def chat():
    # Chat could be free as per "except research all are for free"
    # But usually it's part of the research dashboard.
    message = request.json.get('message')
    history = session.get('tumtum_history', [])
    response_text, new_history = generate_tumtum_chat(message, history)
    session['tumtum_history'] = new_history
    return jsonify({'success': True, 'response': response_text})
