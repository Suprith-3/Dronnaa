import os
import shutil

# Path where I generated the high-quality logo v3
source_path = r"C:\Users\supre\.gemini\antigravity\brain\c739a058-187b-4974-b6e6-e608545aa602\dronacharya_hub_logo_v3_1775051264925.png"

# Target locations
targets = [
    os.path.join('static', 'logo.png'),
    os.path.join('app', 'static', 'logo.png')
]

print("--- DEPLOYING OFFICIAL LOGO ---")

if not os.path.exists(source_path):
    print(f"❌ Source file not found: {source_path}")
else:
    for target in targets:
        folder = os.path.dirname(target)
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        try:
            shutil.copy2(source_path, target)
            print(f"✅ Deployed to: {os.path.abspath(target)}")
        except Exception as e:
            print(f"❌ Failed to deploy to {target}: {e}")

print("-------------------------------")
print("1. Refresh your browser (Ctrl+F5)")
print("2. If it's still broken, RESTART the server (Ctrl+C and run python app.py)")
