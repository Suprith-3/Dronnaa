from . import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
import uuid, json, math

# ─────────────────────────────────────────
# TUTOR MODELS
# ─────────────────────────────────────────

user_video_completion = db.Table('user_video_completion',
    db.Column('user_id',  db.Integer, db.ForeignKey('user.id'),  primary_key=True),
    db.Column('video_id', db.Integer, db.ForeignKey('video.id'), primary_key=True)
)

class User(UserMixin, db.Model):
    __table_args__ = {'extend_existing': True}
    id           = db.Column(db.Integer, primary_key=True)
    first_name   = db.Column(db.String(80),  nullable=False)
    last_name    = db.Column(db.String(80),  nullable=False)
    username     = db.Column(db.String(80),  unique=True, nullable=False)
    dob          = db.Column(db.String(20))
    email        = db.Column(db.String(120), unique=True, nullable=False)
    profile_pic  = db.Column(db.String(120), default='default.png')
    password_hash= db.Column(db.String(200), nullable=False)
    has_research_access = db.Column(db.Boolean, default=False)
    completed_videos = db.relationship('Video', secondary=user_video_completion,
                                       lazy='dynamic', backref=db.backref('completed_by_users', lazy=True))
    quiz_history = db.relationship('QuizHistory', backref='user', lazy=True, foreign_keys='QuizHistory.user_id')
    coupons      = db.relationship('UserCoupon',  backref='user', lazy=True)

    def __init__(self, **kwargs):
        raw_password = kwargs.pop('password', None)
        if 'username' not in kwargs or not kwargs['username']:
            name = kwargs.get('first_name', 'user')
            kwargs['username'] = f"{''.join(c for c in name.lower() if c.isalnum())}_{str(uuid.uuid4())[:8]}"
        super().__init__(**kwargs)
        if raw_password:
            self.set_password(raw_password)

    def set_password(self, p): self.password_hash = generate_password_hash(p)
    def check_password(self, p): return check_password_hash(self.password_hash, p)

class Course(db.Model):
    __table_args__ = {'extend_existing': True}
    id          = db.Column(db.Integer, primary_key=True)
    title       = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, default='')
    level       = db.Column(db.String(30), default='Beginner')
    is_permanent = db.Column(db.Boolean, default=False)
    videos      = db.relationship('Video', backref='course', lazy=True)
    quizzes     = db.relationship('Quiz',  backref='course', lazy=True)

class Video(db.Model):
    __table_args__ = {'extend_existing': True}
    id        = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    title     = db.Column(db.String(200), nullable=False)
    video_url = db.Column(db.String(300), nullable=False)

class Quiz(db.Model):
    __table_args__ = {'extend_existing': True}
    id        = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    question  = db.Column(db.Text, nullable=False)
    answer    = db.Column(db.String(200), nullable=False)

class QuizHistory(db.Model):
    __table_args__ = {'extend_existing': True}
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    topic           = db.Column(db.String(200), default='')
    score           = db.Column(db.Integer,  default=0)
    total_questions = db.Column(db.Integer,  default=10)
    percentage      = db.Column(db.Float,    default=0.0)
    date_taken      = db.Column(db.DateTime, default=datetime.utcnow)
    details         = db.Column(db.Text,     default='[]')  # JSON list of Q&A

# ─────────────────────────────────────────
# MY BOOK MODEL
# ─────────────────────────────────────────

class Book(db.Model):
    __table_args__ = {'extend_existing': True}
    id            = db.Column(db.Integer, primary_key=True)
    title         = db.Column(db.String(200), nullable=False)
    price         = db.Column(db.Float, default=0.0)
    cover_image   = db.Column(db.String(200), nullable=True)
    seller_name   = db.Column(db.String(100), nullable=False)
    seller_phone  = db.Column(db.String(20),  nullable=False)
    latitude      = db.Column(db.Float, nullable=False)
    longitude     = db.Column(db.Float, nullable=False)
    address       = db.Column(db.String(300), default='')
    genre         = db.Column(db.String(80),  default='General')
    condition     = db.Column(db.String(30),  default='Good')
    description   = db.Column(db.Text, default='')
    listing_type  = db.Column(db.String(20),  default='sell')  # sell / swap / donate
    is_available  = db.Column(db.Boolean, default=True)
    date_listed   = db.Column(db.DateTime, default=datetime.utcnow)
    views         = db.Column(db.Integer, default=0)
    user_id       = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # owner
    user          = db.relationship('User', backref=db.backref('books', lazy=True))


# ─────────────────────────────────────────
# NOTES MODEL
# ─────────────────────────────────────────

class Note(db.Model):
    __table_args__ = {'extend_existing': True}
    id            = db.Column(db.Integer, primary_key=True)
    title         = db.Column(db.String(200), nullable=False)
    subject       = db.Column(db.String(100), nullable=False)
    description   = db.Column(db.Text, default='')
    file_path     = db.Column(db.String(200), nullable=False)
    uploader_name = db.Column(db.String(100), nullable=False)
    date_uploaded = db.Column(db.DateTime, default=datetime.utcnow)
    user_id       = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # owner
    is_handwritten = db.Column(db.Boolean, default=False)
    user          = db.relationship('User', backref=db.backref('notes', lazy=True))


# ─────────────────────────────────────────
# COUPON MODEL
# ─────────────────────────────────────────

class UserCoupon(db.Model):
    __table_args__ = {'extend_existing': True}
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    coupon_url    = db.Column(db.String(300), nullable=False)
    date_earned   = db.Column(db.DateTime, default=datetime.utcnow)

# ─────────────────────────────────────────
# DISTANCE HELPER
# ─────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))
