import os, urllib.parse, uuid
from flask import render_template, request, redirect, url_for, jsonify, current_app, flash
from werkzeug.utils import secure_filename
from . import books_bp
from ..models import Book, haversine_km, db

@books_bp.route('/')
def index():
    # Featured books
    books = Book.query.filter_by(is_available=True).order_by(Book.date_listed.desc()).limit(3).all()
    return render_template('books/index.html', books=books)

@books_bp.route('/browse')
def browse():
    books = Book.query.filter_by(is_available=True).order_by(Book.date_listed.desc()).all()
    return render_template('books/browse.html', books=books)

@books_bp.route('/book/<int:id>')
def book_detail(id):
    book = Book.query.get_or_404(id)
    book.views += 1
    db.session.commit()
    wa_msg = f"Hi {book.seller_name}, I saw your book '{book.title}' on Dronacharya Hub. Is it still available?"
    wa_url = f"https://wa.me/91{book.seller_phone.replace(' ','').replace('-','')}?text={urllib.parse.quote(wa_msg)}"
    return render_template('books/book_detail.html', book=book, wa_url=wa_url)

@books_bp.route('/list', methods=['GET', 'POST'])
def list_book():
    if request.method == 'POST':
        title = request.form.get('title')
        price = float(request.form.get('price', 0))
        seller_name = request.form.get('seller_name')
        seller_phone = request.form.get('seller_phone')
        latitude = float(request.form.get('latitude', 0))
        longitude = float(request.form.get('longitude', 0))
        address = request.form.get('address')
        genre = request.form.get('genre', 'General')
        condition = request.form.get('condition', 'Good')
        description = request.form.get('description', '')
        listing_type = request.form.get('listing_type', 'sell')
        
        filename = 'default_book.png'
        if 'cover_image' in request.files:
            file = request.files['cover_image']
            if file and file.filename != '':
                print(f"DEBUG: Uploading cover image: {file.filename}")
                from modules.supabase_helper import upload_file_to_supabase
                s_url = upload_file_to_supabase(file, folder="books")
                if s_url:
                    print(f"DEBUG: Successfully uploaded to Supabase: {s_url}")
                    filename = s_url
                else:
                    print(f"DEBUG: Supabase upload failed, falling back to local storage")
                    filename = secure_filename(f"book_{uuid.uuid4().hex}_{file.filename}")
                    try:
                        file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
                        print(f"DEBUG: Saved locally: {filename}")
                    except Exception as e:
                        print(f"DEBUG: Local save failed: {e}")
                        filename = 'default_book.png'
        
        print(f"DEBUG: Creating new book with image: {filename}")
        new_book = Book(
            title=title, price=price, seller_name=seller_name, seller_phone=seller_phone,
            latitude=latitude, longitude=longitude, address=address,
            genre=genre, condition=condition, description=description,
            listing_type=listing_type, cover_image=filename
        )
        db.session.add(new_book)
        db.session.commit()
        print(f"DEBUG: Book '{title}' committed to database.")
        flash('Book listed successfully!', 'success')
        return redirect(url_for('books.browse'))
    return render_template('books/list_book.html')

@books_bp.route('/search')
def search():
    q = request.args.get('q', '').strip()
    books = Book.query.filter(Book.title.ilike(f'%{q}%'), Book.is_available==True).all()
    return render_template('books/search_results.html', books=books, q=q)

@books_bp.route('/mark-sold/<int:id>', methods=['POST'])
def mark_sold(id):
    book = Book.query.get_or_404(id)
    book.is_available = False
    db.session.commit()
    return jsonify({'success': True})

@books_bp.route('/api/books')
def api_all_books():
    books = Book.query.filter_by(is_available=True).all()
    return jsonify([{
        'id': b.id, 'title': b.title, 'latitude': b.latitude, 'longitude': b.longitude
    } for b in books])

@books_bp.route('/api/nearby')
def api_nearby():
    try:
        lat = float(request.args.get('lat', 0))
        lon = float(request.args.get('lon', 0))
        radius = float(request.args.get('radius', 10))
    except:
        return jsonify([])
    
    q = request.args.get('q', '').strip()

    books_query = Book.query.filter_by(is_available=True)
    if q:
        books_query = books_query.filter(Book.title.ilike(f'%{q}%'))
    
    books = books_query.all()
    nearby = []
    for b in books:
        dist = haversine_km(lat, lon, b.latitude, b.longitude)
        if dist <= radius:
            wa_msg = f"Hi {b.seller_name}, I saw your book '{b.title}' on Dronacharya Hub. Is it still available?"
            nearby.append({
                'id': b.id, 'title': b.title, 'price': b.price,
                'seller_name': b.seller_name,
                'seller_phone': b.seller_phone,
                'cover_image': b.cover_image,
                'latitude': b.latitude, 'longitude': b.longitude,
                'address': b.address,
                'listing_type': b.listing_type,
                'distance_km': round(dist, 2),
                'distance_label': f"{int(dist*1000)}m away" if dist < 1 else f"{dist:.1f}km away",
                'wa_url': f"https://wa.me/91{b.seller_phone.replace(' ','').replace('-','')}?text={urllib.parse.quote(wa_msg)}",
                'url': url_for('books.book_detail', id=b.id)
            })
    nearby.sort(key=lambda x: x['distance_km'])
    return jsonify(nearby)
