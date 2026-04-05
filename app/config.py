import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dronacharya-hub-secret-2024')

    # Database settings
    database_url = os.getenv('DATABASE_URL', 'sqlite:///dronacharya.db')
    if database_url and database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    SQLALCHEMY_DATABASE_URI = database_url
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Supabase settings for Storage
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')

    # Folder/Bucket settings
    UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'static', 'uploads')
    AUDIO_FOLDER  = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'static', 'audio')
    DOCS_FOLDER   = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'static', 'documents')
    NOTES_FOLDER  = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'static', 'notes')
    TEMP_FOLDER   = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'temp_files')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'pdf'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

