import os
from supabase import create_client, Client
from flask import current_app

def get_supabase_client() -> Client:
    url = current_app.config.get('SUPABASE_URL')
    key = current_app.config.get('SUPABASE_KEY')
    if not url or not key:
        return None
    return create_client(url, key)

def upload_file_to_supabase(bucket: str, file_obj, remote_path: str):
    """
    Uploads a file object to Supabase storage.
    Returns the public URL of the uploaded file.
    """
    supabase = get_supabase_client()
    if not supabase:
        print("DEBUG: Supabase client not initialized (check ENV).")
        return None

    try:
        # Seek to start just in case
        file_obj.seek(0)
        file_data = file_obj.read()
        
        # Determine content type (basic check)
        content_type = "application/octet-stream"
        if remote_path.endswith('.png'): content_type = "image/png"
        elif remote_path.endswith('.jpg') or remote_path.endswith('.jpeg'): content_type = "image/jpeg"
        elif remote_path.endswith('.pdf'): content_type = "application/pdf"
        elif remote_path.endswith('.docx'): content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif remote_path.endswith('.mp3'): content_type = "audio/mpeg"

        res = supabase.storage.from_(bucket).upload(
            path=remote_path,
            file=file_data,
            file_options={"content-type": content_type}
        )
        
        # Get public URL
        url_res = supabase.storage.from_(bucket).get_public_url(remote_path)
        # Handle various response formats (some return objects with .public_url or strings)
        if hasattr(url_res, 'public_url'):
            return url_res.public_url
        if isinstance(url_res, dict) and 'publicUrl' in url_res:
            return url_res['publicUrl']
        return str(url_res)
    except Exception as e:
        import traceback
        print(f"DEBUG: Supabase upload failed for {remote_path}: {str(e)}")
        traceback.print_exc()
        return None

def delete_file_from_supabase(bucket: str, remote_path: str):
    supabase = get_supabase_client()
    if not supabase: return False
    try:
        supabase.storage.from_(bucket).remove([remote_path])
        return True
    except:
        return False
