import os
from supabase import create_client, Client
from flask import current_app

def get_supabase_client() -> Client:
    url = current_app.config.get('SUPABASE_URL')
    key = current_app.config.get('SUPABASE_KEY')
    if not url or not key:
        return None
    return create_client(url, key)

def upload_file_to_supabase(file_obj, folder="uploads", bucket="dronacharya-hub"):
    """
    Uploads a file object to Supabase Storage and returns the public URL.
    """
    supabase = get_supabase_client()
    if not supabase:
        return None

    try:
        # Seek to start of file to ensure full upload
        file_obj.seek(0)
        file_data = file_obj.read()
        
        # Create a unique filename
        import uuid
        ext = 'png'
        if hasattr(file_obj, 'filename') and '.' in file_obj.filename:
            ext = file_obj.filename.split('.')[-1]
        
        filename = f"{folder}/{uuid.uuid4().hex}.{ext}"
        
        # Upload
        content_type = getattr(file_obj, 'content_type', 'image/png')
        
        # In newer version of supabase-py, upload might raise or return response
        res = supabase.storage.from_(bucket).upload(
            path=filename,
            file=file_data,
            file_options={"content-type": content_type}
        )
        
        # Check if response indicates an error (depends on SDK version)
        # If it's a response object, it might have an error attribute or status code
        if hasattr(res, 'error') and res.error:
            print(f"Supabase Upload Error response: {res.error}")
            return None
            
        # Get public URL
        # URL format: {SUPABASE_URL}/storage/v1/object/public/{bucket}/{filename}
        public_url = f"{current_app.config['SUPABASE_URL']}/storage/v1/object/public/{bucket}/{filename}"
        print(f"Uploaded to Supabase: {public_url}")
        return public_url
    except Exception as e:
        print(f"Supabase Storage Exception: {e}")
        return None
