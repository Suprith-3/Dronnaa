import os
import re
import traceback
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required
from google import genai

animated_learning_bp = Blueprint('animated_learning', __name__)

@animated_learning_bp.route('/')
@login_required
def index():
    return render_template('animated_learning/index.html')

@animated_learning_bp.route('/generate', methods=['POST'])
@login_required
def generate():
    try:
        # 1. Parse JSON
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid request - JSON required"}), 400

        user_prompt = data.get('prompt')
        api_key = os.getenv('GEMINI_API_KEY')

        if not user_prompt:
            return jsonify({"error": "Prompt is required"}), 400
        if not api_key:
            return jsonify({"error": "Gemini API Key (GEMINI_API_KEY) not found in server environment"}), 500

        print(f"Animated Learning (Native): Thinking about '{user_prompt[:30]}...'")

        # Initialize Google GenAI Client
        client = genai.Client(api_key=api_key)

        # 2. Brain Phase (Thinking & Explanation Only)
        # Using Gemini 1.5 Flash as it has high free tier limits for text
        brain_prompt = f"Explain this educational concept thoroughly but simply: {user_prompt}. Provide a clear, engaging explanation for a student. Format your response exactly like this: [EXPLANATION: explanation text]"
        
        try:
            brain_response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=brain_prompt
            )
            
            brain_text = brain_response.text
            expl_match = re.search(r'\[EXPLANATION: (.*?)\]', brain_text, re.S)
            explanation = expl_match.group(1).strip() if expl_match else brain_text
            
        except Exception as e:
            print(f"Animated Learning ERROR: Brain reasoning failed: {str(e)}")
            return jsonify({"error": "AI Reasoning is currently hitting a quota limit. Please try again in 60 seconds."}), 429

        return jsonify({
            "status": "success",
            "video_url": None, # Video generation removed as requested
            "explanation": explanation
        })

    except Exception as e:
        print(f"Animated Learning CRASH: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
