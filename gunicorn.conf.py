import os

# Port settings for Render
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Gunicorn setup (Tuned for faster response)
timeout = 120
workers = 2 
preload_app = False 
