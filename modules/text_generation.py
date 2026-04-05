import os, json, random
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

def serialize_history(history):
    result = []
    for h in history:
        if hasattr(h, 'role') and hasattr(h, 'parts'):
            result.append({'role': h.role, 'parts': [{'text': str(p.text) if hasattr(p, 'text') else str(p)} for p in h.parts]})
        else:
            result.append(h)
    return result

def generate_gemini_quiz(topic):
    system_instruction = (
        f"You are an expert teacher. Generate exactly 10 multiple-choice questions about: '{topic}'. "
        f"Return ONLY valid JSON (no markdown, no extra text): "
        f'[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"answer":"A"}}]'
    )
    try:
        response = model.generate_content(system_instruction)
        text = response.text.strip()
        if '```json' in text: text = text.split('```json')[1].split('```')[0].strip()
        return json.loads(text)
    except Exception as e:
        err_msg = str(e)
        print(f"Gemini Quiz Error: {err_msg}")
        if "429" in err_msg or "quota" in err_msg.lower():
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
    chat = model.start_chat(history=history)
    try:
        response = chat.send_message(message)
        return response.text, serialize_history(chat.history)
    except Exception as e:
        print(f"Gemini Chat Error: {e}")
        return "I am currently unable to answer that. Please try again later.", history

def generate_tumtum_chat(message, history):
    chat = model.start_chat(history=history)
    try:
        response = chat.send_message(message)
        return response.text, serialize_history(chat.history)
    except Exception as e:
        err_msg = str(e)
        print(f"TumTum API Error: {err_msg}")
        return f"AI Assistant Error: {err_msg}", history

def generate_gemini_paper(topic, language):
    system_instruction = (
        f"You are an expert academic writer. Generate a comprehensive research paper about '{topic}' "
        f"written entirely in {language}. Include these exact markdown sections: "
        f"## Research Topic\n## Abstract\n## Introduction\n## Literature Review\n"
        f"## Methodology\n## Main Findings / Discussion\n## Conclusion\n## References\n"
        f"Under References, list exactly 12 realistic academic references. "
        f"Respond ONLY with markdown content. No preamble."
    )
    try:
        response = model.generate_content(system_instruction)
        if response.candidates and len(response.candidates) > 0:
            if response.candidates[0].finish_reason == 3:
                return "ERROR: Restricted by safety filters."
            return response.text
        return "ERROR: No response from AI."
    except Exception as e:
        err_msg = str(e)
        print(f"Gemini Research Error: {err_msg}")
        if "429" in err_msg or "quota" in err_msg.lower():
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
        f"You are a professional librarian and academic citation expert. "
        f"Generate a high-precision, official {style} style citation for the provided source: '{source}'. "
        f"{context_info}"
        f"\n\nRules:"
        f"\n1. If a URL is given, use your knowledge to find the correct author, full paper title, publication year, and journal name."
        f"\n2. If only a short snippet is provided, format it professionally based on the {style} manual."
        f"\n3. Respond ONLY with the formatted citation string. No preamble, no conversational text."
    )
    try:
        response = model.generate_content(system_instruction)
        return response.text.strip()
    except Exception as e:
        err_msg = str(e)
        print(f"Gemini Citation Error: {err_msg}")
        if "429" in err_msg or "quota" in err_msg.lower():
            # Demo fallback
            return f"{source} ({style} style). [Note: Quota Exceeded, using simplified format]"
        return None

def generate_gemini_notes(topic):
    try:
        response = model.generate_content(f"Generate study notes for: {topic}")
        if response.text:
            return response.text
        return f"# Study Notes: {topic}\n(Empty response from AI)"
    except Exception as e:
        print(f"Gemini Notes Error: {e}")
        return f"# Study Notes: {topic}\n(Error)"

def generate_chess_move(fen):
    system_instruction = (
        f"You are a professional Grandmaster Chess Engine. The current board state in FEN is: '{fen}'. "
        f"Choose the absolute best UCI move (e.g., 'e2e4') for the player whose turn it is. "
        f"Return ONLY the UCI move string. No explanation, no text."
    )
    try:
        response = model.generate_content(system_instruction)
        return response.text.strip().lower()
    except Exception as e:
        print(f"Gemini Chess Move Error: {e}")
        return None

def generate_crossmath_puzzle(difficulty="Medium"):
    system_instruction = (
        f"Generate a {difficulty} Crossmath puzzle (math grid). Return ONLY valid JSON. "
        f"Format: {{\"grid\": [[\"5\",\"+\",\"3\",\"=\",\"8\"],[\"*\",\" \",\" \",\" \",\"+\"],[\"2\",\"+\",\"4\",\"=\",\"6\"],[\"=\",\" \",\" \",\" \",\"=\"],[\"10\",\" \",\" \",\" \",\"14\"]], \"clues\": [\"Row 1: 5+3=8\", \"Col 1: 5*2=10\"]}} "
        f"Ensure the math is correct."
    )
    try:
        response = model.generate_content(system_instruction)
        text = response.text.strip()
        if '```json' in text: text = text.split('```json')[1].split('```')[0].strip()
        return json.loads(text)
    except Exception as e:
        err_msg = str(e)
        print(f"Gemini Crossmath Error: {err_msg}")
        if "429" in err_msg or "quota" in err_msg.lower() or "limit" in err_msg.lower():
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
