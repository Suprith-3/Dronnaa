import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_cors import CORS
from dotenv import load_dotenv
from .config import Config

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
login_manager.login_view = 'tutor.login'

def create_app():
    load_dotenv()
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.config.from_object(Config)

    # Create all required folders
    for folder in [app.config['UPLOAD_FOLDER'], app.config['AUDIO_FOLDER'],
                   app.config['DOCS_FOLDER'], app.config['NOTES_FOLDER'], app.config['TEMP_FOLDER']]:
        os.makedirs(folder, exist_ok=True)

    CORS(app)
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)

    with app.app_context():
        from .models import User
        @login_manager.user_loader
        def load_user(user_id): return User.query.get(int(user_id))

        from . import models

        # Register all 6 blueprints
        from .tutor.routes   import tutor_bp
        from .research.routes import research_bp
        from .books.routes   import books_bp
        from .scholarships.routes import scholarship_bp
        from .notes.routes   import notes_bp
        from .games.routes   import games_bp

        app.register_blueprint(tutor_bp,    url_prefix='/tutor')
        app.register_blueprint(research_bp, url_prefix='/research')
        app.register_blueprint(books_bp,    url_prefix='/books')
        app.register_blueprint(scholarship_bp, url_prefix='/scholarships')
        app.register_blueprint(notes_bp,    url_prefix='/notes')
        app.register_blueprint(games_bp,    url_prefix='/games')

        # Homepage route
        from flask import render_template
        @app.route('/')
        def home():
            return render_template('home.html')

        db.create_all()

        # Auto-seed the database if it's currently empty (fresh deployment)
        from .models import Course
        if Course.query.first() is None:
            from .seed_helper import auto_seed_courses
            auto_seed_courses(db)

    return app

# Add this to handle Render's default 'gunicorn app:app' command
# It creates a global instance that is only exported when the package is imported
if os.environ.get('RENDER'):
    app = create_app()
