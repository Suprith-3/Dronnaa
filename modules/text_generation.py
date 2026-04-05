import os, json, random
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
model_id = 'gemini-flash-latest' # Stable alias for 1.5-flash or 2.0-flash depending on region
fallback_models = ['gemini-flash-lite-latest', 'gemini-pro-latest', 'gemini-2.0-flash']




def serialize_history(history):
    if history is None: return []
    result = []
    for h in history:
        # Compatibility check for both SDK types
        if hasattr(h, 'role') and hasattr(h, 'parts'):
            parts = []
            for p in h.parts:
                if hasattr(p, 'text'): parts.append({'text': str(p.text)})
                elif isinstance(p, dict) and 'text' in p: parts.append(p)
                else: parts.append({'text': str(p)})
            result.append({'role': h.role, 'parts': parts})
        elif isinstance(h, dict) and 'role' in h:
            result.append(h)
    return result

def generate_gemini_quiz(topic):
    system_instruction = (
        f"You are an expert teacher. Generate exactly 10 multiple-choice questions about: '{topic}'. "
        f"Return ONLY valid JSON (no markdown, no extra text): "
        f'[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"answer":"A"}}]'
    )
    try:
        response = client.models.generate_content(model=model_id, contents=system_instruction)
        text = response.text.strip()
        if '```json' in text: text = text.split('```json')[1].split('```')[0].strip()
        return json.loads(text)
    except Exception as e:
        err_msg = str(e)
        print(f"Gemini Quiz Error: {err_msg}")
        # Always try fallback for ANY error that prevents generation
        if True:
            # Try fallbacks with a tiny delay
            import time
            for fb_model in fallback_models:
                try:
                    time.sleep(1) # Tiny pause to avoid spamming
                    response = client.models.generate_content(model=fb_model, contents=system_instruction)
                    text = response.text.strip()
                    if '```json' in text: text = text.split('```json')[1].split('```')[0].strip()
                    return json.loads(text)
                except:
                    continue

            return [
                {"question": f"Which of the following describes a core aspect of {topic}?", "options": {"A": "Option 1", "B": "Option 2", "C": "Correct Principle", "D": "None"}, "answer": "C"},
                {"question": f"True or False: {topic} is widely used in modern education.", "options": {"A": "True", "B": "False", "C": "Maybe", "D": "Not sure"}, "answer": "A"}
            ]
        return []

def generate_gemini_chat(message, history):
    system_instruction = (
        "You are Dronacharya, the legendary wise teacher from ancient India, now an AI tutor. "
        "Be patient, clear, encouraging. Use examples. Format responses in markdown. "
        "Never refuse a learning question."
    )
    try:
        chat = client.chats.create(model=model_id, config={'system_instruction': system_instruction}, history=history)
        response = chat.send_message(message)
        # Fix: Ensure history is retrieved safely
        try:
            return response.text, serialize_history(chat.history)
        except AttributeError:
            # Fallback for SDK version differences
            return response.text, history
    except Exception as e:
        err_msg = str(e)
        print(f"Gemini Chat Error: {err_msg}")
        if True:
            import time
            for fb_model in fallback_models:
                try:
                    time.sleep(1)
                    chat = client.chats.create(model=fb_model, config={'system_instruction': system_instruction}, history=history)
                    response = chat.send_message(message)
                    return response.text, serialize_history(chat.history)
                except:
                    continue

        return "I am currently unable to answer that. Please try again later.", history

def generate_tumtum_chat(message, history):
    system_instruction = (
        "You are TumTum, the Research Assistant for Dronacharya Hub. "
        "Your goal is to help students with academic research, brainstorming, and writing. "
        "You must advocate for and explain these 10 Research Quality Rules:\n"
        "1. Always Cite Sources (Quotes, paraphrases, data).\n"
        "2. Minimal Quotes (<10% of paper, max 2 sentences).\n"
        "3. Authentic Paraphrasing (Complete rewording, not synonym swapping).\n"
        "4. Original Voice (85-90% original analysis).\n"
        "5. Citation Breadth (Cite unique ideas, not common knowledge).\n"
        "6. Consistent Style (APA/MLA/etc).\n"
        "7. Full Bibliography (Alphabetized, matching in-text).\n"
        "8. No Replication (Unique synthesis per request).\n"
        "9. Low Plagiarism (<10% similarity goal).\n"
        "10. Source Documentation (Keep track of Author, Title, Date, URL).\n\n"
        "Be helpful, academic, yet encouraging. Keep responses concise unless asked for detail."
    )
    try:
        chat = client.chats.create(model=model_id, config={'system_instruction': system_instruction}, history=history)
        response = chat.send_message(message)
        try:
            return response.text, serialize_history(chat.history)
        except AttributeError:
            return response.text, history
    except Exception as e:
        err_msg = str(e)
        print(f"TumTum API Error: {err_msg}")
        if True:
            import time
            for fb_model in fallback_models:
                try:
                    time.sleep(1)
                    chat = client.chats.create(model=fb_model, config={'system_instruction': system_instruction}, history=history)
                    response = chat.send_message(message)
                    return response.text, serialize_history(chat.history)
                except:
                    continue

        return f"AI Assistant Error: {err_msg}", history

def generate_gemini_paper(topic, language):
    system_instruction = (
        f"You are a PhD-level academic writer. Generate a comprehensive and extensive research paper about '{topic}' "
        f"written entirely in {language}. The paper MUST be between 2000 and 2500 words in length. "
        f"Each section should be highly detailed with in-depth academic analysis. "
        f"You MUST strictly adhere to these 10 Academic Quality Rules:\n"
        f"1. CITE ALL SOURCES: Every direct quote, paraphrased idea, and data point/finding must have an in-text citation.\n"
        f"2. MINIMAL QUOTES: Direct quotes must be <10% of the paper. Each quote max 2 sentences. Include page numbers if applicable.\n"
        f"3. AUTHENTIC PARAPHRASING: Reword source material completely with new structure. Do not just swap synonyms. Citations still required.\n"
        f"4. ORIGINAL VOICE: 85-90% of the paper must be your original analysis, interpretation, and synthesis of ideas. Lead with your own perspective.\n"
        f"5. CITATION BREADTH: Cite unique theories, data, and findings. Common knowledge does not need citation.\n"
        f"6. STYLE CONSISTENCY: Follow the chosen style (APA 7th, MLA 9th, Chicago 17th, IEEE, Harvard, or Vancouver) exactly as per current standards.\n"
        f"7. COMPLETE BIBLIOGRAPHY: Every source cited in-text must appear in an alphabetized References section at the end with proper hanging indents.\n"
        f"8. NO REPLICATION: Content must be unique and specifically tailored to this prompt. No bulk reuse of generic material.\n"
        f"9. LOW PLAGIARISM: Ensure the content would pass a Turnitin check with <10% similarity score by being highly original in synthesis.\n"
        f"10. CITATION CRITERIA: Ensure each reference includes Author, Year/Date, Title, Journal/Proceedings, Volume(Issue), Page Numbers, and DOI/URL. "
        f"Identify the source type (Journal, Book, Conference, etc.) and apply the universal citation criteria for maximum accuracy and verifiability.\n\n"
        f"Include these exact markdown sections:\n"
        f"## Research Topic\n## Abstract\n## Introduction\n## Literature Review\n"
        f"## Theoretical Framework\n## Methodology\n## Main Findings / Discussion\n"
        f"## Conclusion\n## References\n"
        f"Under References, list exactly 12 realistic academic references. "
        f"Maintain a scholarly tone. Respond ONLY with markdown content. No preamble."
    )
    try:
        response = client.models.generate_content(
            model=model_id, 
            contents=system_instruction,
            config={'max_output_tokens': 8192, 'temperature': 0.7}
        )
        if response.candidates and len(response.candidates) > 0:
            if response.candidates[0].finish_reason == 3:
                return "ERROR: Restricted by safety filters."
            return response.text
        return "ERROR: No response from AI."
    except Exception as e:
        err_msg = str(e)
        print(f"Gemini Research Error: {err_msg}")
        if True:
            # Try fallback models for real paper generation
            for fb_model in fallback_models:
                try:
                    fb_response = client.models.generate_content(
                        model=fb_model, 
                        contents=system_instruction,
                        config={'max_output_tokens': 8192, 'temperature': 0.7}
                    )
                    if fb_response.text:
                        return fb_response.text
                except:
                    continue
            
            # If all fail, use demo mode
            refs = "\n".join([f"{i}. Expert, A. (202{random.randint(0,5)}). Research on {topic} Vol {i}." for i in range(1, 13)])
            return f"""# [DEMO MODE: Quota Exceeded] Research Paper: {topic}
## Abstract
This paper explores {topic}. Due to AI quota limits, this sample serves as a placeholder for testing.

## Introduction
The study of {topic} is crucial...

## Methodology
Structured analysis was conducted...

## Conclusion
Final results indicate clear trends...

## References
{refs}"""
        return f"ERROR: {err_msg}"

def generate_gemini_citation(source, style):
    # If the source is a URL, try to enrichment with title fetching
    context_info = ""
    if source.startswith("http") or "www." in source:
        try:
            import urllib.request
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(source, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                html = response.read().decode('utf-8', errors='ignore')
                import re
                title_match = re.search('<title>(.*?)</title>', html, re.IGNORECASE)
                if title_match:
                    context_info = f" [Fetched Context: {title_match.group(1).strip()}]"
        except:
            pass
            
    system_instruction = (
        f"You are a professional librarian and PhD-level academic citation expert. "
        f"Generate a high-precision, official {style} style citation for the provided source: '{source}'. "
        f"{context_info}"
        f"\n\n### MANDATORY UNIVERSAL CRITERIA:"
        f"\n1. ACCURACY: Verify all information. Identify if the source is a Journal, Conference Paper, Book, Thesis, or Website."
        f"\n2. COMPLETENESS: Include Author(s), Year, Title, Source/Container, Volume/Issue, Page Numbers, and DOI/URL as per {style} standards."
        f"\n3. FORMATTING: Adhere strictly to {style} version (APA 7th, MLA 9th, Chicago 17th, etc.)."
        f"\n\n### STYLE-SPECIFIC RULES TO OBEY:"
        f"\n- APA (7th): (Author, Year). Title. Journal, Vol(Issue), pp. DOI."
        f"\n- MLA (9th): Author. 'Title.' Journal, vol., no., Year, pp. DOI/URL."
        f"\n- CHICAGO: Author, Title (Publisher, Year), Page."
        f"\n- IEEE: [#] Initials. Surname, 'Title,' Journal, vol. #, no. #, pp., Month Year, doi: DOI."
        f"\n- HARVARD: Author (Year) 'Title', Journal, vol(issue), pp."
        f"\n- VANCOUVER: Author Surname AF. Title. Abbr Journal. Year;Vol(Issue):pp. DOI."
        f"\n\n### CRITERIA BY TYPE:"
        f"\n- For Journals: Must include Volume, Issue, and inclusive Page Numbers."
        f"\n- For Conferences: Include Proceedings name, location (if applicable), and Publisher."
        f"\n- For Books: Include Publisher and Edition (if not 1st)."
        f"\n- For Websites: Use Organization name if author unknown; include URL and Access Date."
        f"\n\nRespond ONLY with the formatted citation string. No preamble, no conversational text."
    )
    try:
        response = client.models.generate_content(model=model_id, contents=system_instruction)
        return response.text.strip()
    except Exception as e:
        err_msg = str(e)
        print(f"Gemini Citation Error: {err_msg}")
        if True:
            for fb_model in fallback_models:
                try:
                    fb_response = client.models.generate_content(model=fb_model, contents=system_instruction)
                    if fb_response.text:
                        return fb_response.text.strip()
                except:
                    continue
            return f"{source} ({style} style). [Note: Quota Exceeded, using simplified format]"
        return None

def generate_gemini_notes(topic):
    try:
        response = client.models.generate_content(model=model_id, contents=f"Generate study notes for: {topic}")
        if response.text:
            return response.text
        return f"# Study Notes: {topic}\n(Empty response from AI)"
    except Exception as e:
        err_msg = str(e)
        print(f"Gemini Notes Error: {err_msg}")
        if True:
            for fb_model in fallback_models:
                try:
                    fb_response = client.models.generate_content(model=fb_model, contents=f"Generate study notes for: {topic}")
                    if fb_response.text:
                        return fb_response.text
                except:
                    continue
        return f"# Study Notes: {topic}\n(Error: {err_msg})"

def generate_chess_move(fen):
    system_instruction = (
        f"You are a professional Grandmaster Chess Engine. The current board state in FEN is: '{fen}'. "
        f"Choose the absolute best UCI move (e.g., 'e2e4') for the player whose turn it is. "
        f"Return ONLY the UCI move string. No explanation, no text."
    )
    try:
        response = client.models.generate_content(model=model_id, contents=system_instruction)
        return response.text.strip().lower()
    except Exception as e:
        err_msg = str(e)
        print(f"Gemini Chess Move Error: {err_msg}")
        if True:
            for fb_model in fallback_models:
                try:
                    fb_response = client.models.generate_content(model=fb_model, contents=system_instruction)
                    if fb_response.text:
                        return fb_response.text.strip().lower()
                except:
                    continue
        return None

def generate_crossmath_puzzle(difficulty="Medium"):
    system_instruction = (
        f"Generate a {difficulty} Crossmath puzzle (math grid). Return ONLY valid JSON. "
        f"Format: {{\"grid\": [[\"5\",\"+\",\"3\",\"=\",\"8\"],[\"*\",\" \",\" \",\" \",\"+\"],[\"2\",\"+\",\"4\",\"=\",\"6\"],[\"=\",\" \",\" \",\" \",\"=\"],[\"10\",\" \",\" \",\" \",\"14\"]], \"clues\": [\"Row 1: 5+3=8\", \"Col 1: 5*2=10\"]}} "
        f"Ensure the math is correct."
    )
    try:
        response = client.models.generate_content(model=model_id, contents=system_instruction)
        text = response.text.strip()
        if '```json' in text: text = text.split('```json')[1].split('```')[0].strip()
        return json.loads(text)
    except Exception as e:
        err_msg = str(e)
        print(f"Gemini Crossmath Error: {err_msg}")
        if True:
            # Try fallbacks first
            for fb_model in fallback_models:
                try:
                    fb_response = client.models.generate_content(model=fb_model, contents=system_instruction)
                    text = fb_response.text.strip()
                    if '```json' in text: text = text.split('```json')[1].split('```')[0].strip()
                    return json.loads(text)
                except:
                    continue
            
            # Local fallback generator for a valid 3x3 grid
            import random
            def solve(a, op, b):
                if op == '+': return a + b
                if op == '-': return a - b
                if op == '*': return a * b
                return 0
            
            ops = ['+', '-', '*']
            r1_op = random.choice(ops)
            r2_op = random.choice(ops)
            c1_op = random.choice(ops)
            c2_op = random.choice(ops)
            
            a, b = random.randint(1, 10), random.randint(1, 10)
            d, e = random.randint(1, 10), random.randint(1, 10)
            
            # Ensure C1 and C2 don't go negative if using '-'
            if c1_op == '-' and a < d: a, d = d, a
            if c2_op == '-' and b < e: b, e = e, b
            if r1_op == '-' and a < b: a, b = b, a
            if r2_op == '-' and d < e: d, e = e, d

            c = solve(a, r1_op, b)
            f = solve(d, r2_op, e)
            g = solve(a, c1_op, d)
            h = solve(b, c2_op, e)
            
            return {
                "grid": [
                    [str(a), r1_op, str(b), "=", str(c)],
                    [c1_op, " ", " ", " ", " "],
                    [str(d), r2_op, str(e), "=", str(f)],
                    ["=", " ", " ", " ", " "],
                    [str(g), " ", str(h), " ", " "]
                ],
                "clues": [f"Row 1: {a}{r1_op}{b}={c}", f"Col 1: {a}{c1_op}{d}={g}"]
            }
        return None

def generate_gemini_vision(image_path, prompt="Solve this educational problem or explain the handwriting."):
    """Uses Gemini vision to analyze an image (question/problem) and provide an answer."""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            
        # Correct implementation for the latest google-genai SDK to avoid Pydantic validation errors
        response = client.models.generate_content(
            model=model_id,
            contents=[
                types.Content(
                    role='user',
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=image_data, mime_type='image/jpeg')
                    ]
                )
            ]
        )
        return response.text
    except Exception as e:
        err_msg = str(e)
        print(f"Gemini Vision Error: {err_msg}")
        if True:
             for fb_model in fallback_models:
                try:
                    import time
                    time.sleep(1)
                    with open(image_path, 'rb') as f:
                         image_data = f.read()
                    response = client.models.generate_content(
                        model=fb_model,
                        contents=[
                            types.Content(
                                role='user',
                                parts=[
                                    types.Part.from_text(text=prompt),
                                    types.Part.from_bytes(data=image_data, mime_type='image/jpeg')
                                ]
                            )
                        ]
                    )
                    return response.text
                except:
                    continue
        return f"AI Vision Error: {err_msg}"
def generate_gemini_scholarships():
    system_instruction = (
        "You are an educational consultant. Generate exactly 8 real or highly realistic scholarship listings for Indian and international students. "
        "Each scholarship must include: title, agency (providing organization), category (Government, Private, or Non-Government), "
        "eligibility criteria, a brief descriptions, a link (real or official-looking), and the scholarship amount. "
        "The output MUST be a JSON list of objects: "
        '[{"title":"...","agency":"...","category":"...","eligibility":"...","description":"...","link":"...","amount":"...","status":"Open"}]'
        "Return ONLY the valid JSON array."
    )
    try:
        response = client.models.generate_content(model=model_id, contents=system_instruction)
        text = response.text.strip()
        if '```json' in text: text = text.split('```json')[1].split('```')[0].strip()
        return json.loads(text)
    except Exception as e:
        print(f"Gemini Scholarship Error: {e}")
        # Try fallbacks
        for fb_model in fallback_models:
            try:
                fb_response = client.models.generate_content(model=fb_model, contents=system_instruction)
                text = fb_response.text.strip()
                if '```json' in text: text = text.split('```json')[1].split('```')[0].strip()
                return json.loads(text)
            except:
                continue
        # Hardcoded fallback if API fails
        return [
            {
                "title": "Post-Matric Scholarship for OBC Students",
                "agency": "State Government",
                "category": "Government",
                "eligibility": "OBC students with family income < 2.5 Lakh.",
                "description": "Financial assistance for OBC students pursuing post-matriculation courses.",
                "link": "https://scholarships.gov.in/",
                "amount": "Varies by state",
                "status": "Open"
            }
        ]

def generate_gemini_courses():
    system_instruction = (
        "You are an academic curriculum designer. Generate exactly 5 unique and diverse online course ideas for students. "
        "Each course must have: a catchy title, a clear and engaging description (1-2 sentences), and a difficulty level (Beginner, Intermediate, or Advance). "
        "The output MUST be a JSON list of objects: "
        '[{"title":"...","description":"...","level":"..."}]'
        "Return ONLY the valid JSON array."
    )
    try:
        response = client.models.generate_content(model=model_id, contents=system_instruction)
        text = response.text.strip()
        if '```json' in text: text = text.split('```json')[1].split('```')[0].strip()
        data = json.loads(text)
        return data
    except Exception as e:
        print(f"Gemini Course Error: {e}")
        for fb_model in fallback_models:
            try:
                fb_response = client.models.generate_content(model=fb_model, contents=system_instruction)
                text = fb_response.text.strip()
                if '```json' in text: text = text.split('```json')[1].split('```')[0].strip()
                return json.loads(text)
            except:
                continue
        # Hardcoded fallback
        return [
            {"title": "Introduction to Robotics", "description": "Learn the basics of robot design and programming.", "level": "Beginner"},
            {"title": "Modern History of Asia", "description": "Explore the major events that shaped modern Asian nations.", "level": "Intermediate"},
            {"title": "Advanced Data Structures", "description": "Master complex algorithms for efficient data handling.", "level": "Advance"},
            {"title": "Sustainability in Design", "description": "Principles of eco-friendly and ethical design.", "level": "Beginner"},
            {"title": "Particle Physics", "description": "Diving into the fundamental building blocks of the universe.", "level": "Advance"}
        ]

def generate_specific_course(topic):
    system_instruction = (
        f"You are an academic curriculum designer. Generate a single highly relevant and engaging online course idea for the topic: '{topic}'. "
        "The course must have: a catchy title, a clear and engaging description (1-2 sentences), and a difficulty level (Beginner, Intermediate, or Advance). "
        "The output MUST be a JSON object: "
        '{"title":"...","description":"...","level":"..."}'
        "Return ONLY the valid JSON object."
    )
    try:
        response = client.models.generate_content(model=model_id, contents=system_instruction)
        text = response.text.strip()
        if '```json' in text: text = text.split('```json')[1].split('```')[0].strip()
        return json.loads(text)
    except Exception as e:
        print(f"Gemini Specific Course Error: {e}")
        # Try fallbacks
        for fb_model in fallback_models:
            try:
                fb_response = client.models.generate_content(model=fb_model, contents=system_instruction)
                text = fb_response.text.strip()
                if '```json' in text: text = text.split('```json')[1].split('```')[0].strip()
                return json.loads(text)
            except:
                continue
        return {"title": topic.capitalize(), "description": f"A comprehensive course about {topic}.", "level": "Beginner"}
