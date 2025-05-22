from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np
import os
import logging
import base64
import io
from PIL import Image
import time
import uuid
import shutil
import datetime

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inisialisasi Flask
app = Flask(__name__)

# Load model
MODEL_PATH = 'model/model2.keras'
model = load_model(MODEL_PATH)
logger.info(f"Model berhasil dimuat dari: {MODEL_PATH}")

# Mapping kelas (hardcode atau load dari file jika perlu)
class_names = {0: 'ain', 1: 'alif', 2: 'ba', 3: 'dal', 4: 'dhod', 5: 'dzal', 6: 'dzho', 7: 'fa', 8: 'ghoin', 9: 'ha', 10: 'haa', 11: 'hamzah', 12: 'jim', 13: 'kaf', 14: 'kho', 15: 'lam', 16: 'lamalif', 17: 'mim', 18: 'nun', 19: 'qof', 20: 'ro', 21: 'shod', 22: 'sin', 23: 'syin', 24: 'ta', 25: 'tho', 26: 'tsa', 27: 'wawu', 28: 'ya', 29: 'zain'}

# Fungsi preprocessing gambar
def preprocess_image(image, target_size=(156, 156)):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Batch size = 1
    image = image / 255.0  # Normalisasi
    return image

# Fungsi untuk menyimpan sampel gambar untuk debugging
def save_sample_image(image, request_id, request_type, predicted_class=None):
    # Buat direktori samples jika belum ada
    samples_dir = os.path.join('logs', 'samples')
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    
    # Buat nama file dengan timestamp, request_id, dan hasil prediksi
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_text = f"_{predicted_class}" if predicted_class else ""
    filename = f"{timestamp}_{request_id}_{request_type}{result_text}.jpg"
    
    # Simpan gambar
    image_path = os.path.join(samples_dir, filename)
    image.save(image_path)
    
    return image_path

# Fungsi untuk mencatat log request
def log_request(request_type, prediction=None, error=None):
    request_id = str(uuid.uuid4())[:8]  # Generate ID unik untuk request
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    client_ip = request.remote_addr
    
    # Detail request
    request_details = {
        "id": request_id,
        "timestamp": timestamp,
        "ip": client_ip,
        "type": request_type,
        "method": request.method,
        "path": request.path,
        "user_agent": request.headers.get('User-Agent', 'Unknown'),
        "content_type": request.content_type,
        "content_length": request.content_length,
        "headers": {k: v for k, v in request.headers.items()},
        "args": {k: v for k, v in request.args.items()},
        "remote_addr": request.remote_addr,
        "referrer": request.referrer
    }
    
    # Log detail request ke file
    with open(os.path.join('logs', 'detailed_requests.log'), 'a') as f:
        f.write(f"===== REQUEST {request_id} | {timestamp} =====\n")
        f.write(f"IP: {client_ip}\n")
        f.write(f"Type: {request_type}\n")
        f.write(f"Method: {request.method}\n")
        f.write(f"Path: {request.path}\n")
        f.write(f"User-Agent: {request.headers.get('User-Agent', 'Unknown')}\n")
        f.write(f"Content-Type: {request.content_type}\n")
        f.write(f"Content-Length: {request.content_length}\n")
        
        # Log semua header
        f.write("Headers:\n")
        for k, v in request.headers.items():
            f.write(f"  {k}: {v}\n")
        
        # Log query parameters
        f.write("Query Args:\n")
        for k, v in request.args.items():
            f.write(f"  {k}: {v}\n")
        
        # Log form data jika ada
        if request.form:
            f.write("Form Data:\n")
            for k, v in request.form.items():
                # Potong nilai yang panjang
                value_preview = v[:50] + "..." if len(v) > 50 else v
                f.write(f"  {k}: {value_preview}\n")
        
        # Log body request (jika ada dan bukan file)
        if request_type == "json_file_upload":
            try:
                # Simpan JSON
                json_body = request.get_json()
                f.write("Request Body (JSON):\n")
                f.write(f"  {json_body}\n")
            except:
                f.write("Request Body: Unable to parse JSON\n")
        elif request_type == "form_urlencoded_upload":
            try:
                f.write("Request Body (Form URL-Encoded):\n")
                for key in request.form:
                    value = request.form[key]
                    # Potong nilai yang panjang
                    if len(value) > 50:
                        value = value[:50] + "..."
                    f.write(f"  {key}: {value}\n")
            except:
                f.write("Request Body: Unable to parse form data\n")
        elif request_type == "text_upload":
            try:
                # Potong data yang panjang
                text_body = request.data.decode('utf-8')
                text_preview = text_body[:50] + "..." if len(text_body) > 50 else text_body
                f.write("Request Body (Text):\n")
                f.write(f"  {text_preview}\n")
            except:
                f.write("Request Body: Unable to decode text\n")
        elif request_type == "form_data_file" or request_type == "form_data_unnamed":
            f.write("Request Body: Form data with file upload\n")
            try:
                if 'file' in request.files:
                    f.write(f"  File field name: 'file'\n")
                    f.write(f"  Filename: {request.files['file'].filename}\n")
                    f.write(f"  Content-Type: {request.files['file'].content_type}\n")
                elif len(request.files) > 0:
                    first_file = list(request.files.values())[0]
                    f.write(f"  File field name: '{list(request.files.keys())[0]}'\n")
                    f.write(f"  Filename: {first_file.filename}\n")
                    f.write(f"  Content-Type: {first_file.content_type}\n")
            except:
                f.write("  Unable to log file details\n")
        elif request_type == "raw_binary":
            f.write("Request Body: Raw binary data\n")
            f.write(f"  Size: {len(request.data)} bytes\n")
            # Log juga beberapa byte pertama dalam format hex
            hex_preview = ' '.join([f'{b:02x}' for b in request.data[:20]])
            f.write(f"  Hex preview: {hex_preview}...\n")
        
        # Log hasil prediksi atau error
        if prediction:
            f.write("Prediction:\n")
            f.write(f"  Class: {prediction['predicted_class']}\n")
            f.write(f"  Confidence: {prediction['confidence_percent']}\n")
        
        if error:
            f.write(f"Error: {error}\n")
        
        f.write("=" * 50 + "\n\n")
    
    # Log ringkas untuk konsol dan file log utama
    if prediction:
        log_data = {
            "id": request_id,
            "timestamp": timestamp,
            "ip": client_ip,
            "type": request_type,
            "prediction": prediction['predicted_class'],
            "confidence": prediction['confidence_percent']
        }
        logger.info(f"REQUEST #{request_id} | {timestamp} | IP: {client_ip} | Type: {request_type} | " +
                   f"Prediction: {prediction['predicted_class']} | Confidence: {prediction['confidence_percent']}")
    
    if error:
        log_data = {
            "id": request_id,
            "timestamp": timestamp,
            "ip": client_ip,
            "type": request_type,
            "error": str(error)
        }
        logger.error(f"REQUEST #{request_id} | {timestamp} | IP: {client_ip} | Type: {request_type} | Error: {error}")
    
    return request_id, request_details

# Fungsi untuk menentukan nilai kemiripan berdasarkan confidence
def get_similarity(confidence):
    confidence_percent = int(confidence * 100)
    if confidence_percent >= 90:
        return "Sempurna"
    elif confidence_percent >= 70:
        return "Sangat Mirip"
    else:
        return "Mirip"

# Fungsi untuk memformat confidence menjadi skor langsung
def format_confidence(confidence):
    return int(confidence * 100)

# Halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Dokumentasi API
@app.route('/api-docs')
def api_docs():
    return render_template('api_docs.html')

# Endpoint untuk melihat log (dengan autentikasi sederhana)
@app.route('/admin/logs', methods=['GET'])
def view_logs():
    # Autentikasi sederhana dengan parameter query
    admin_key = request.args.get('key', '')
    if admin_key != 'admin123':  # Ganti dengan kunci yang lebih aman di produksi
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Baca file log
        log_path = os.path.join(os.getcwd(), 'logs', 'api.log')
        if not os.path.exists(log_path):
            return jsonify({'error': 'Log file not found'}), 404
        
        # Ambil jumlah baris yang diminta (default 50)
        lines_count = min(int(request.args.get('lines', 50)), 1000)
        
        # Baca baris terakhir dari file log
        with open(log_path, 'r') as f:
            # Baca semua baris dan ambil yang terakhir
            all_lines = f.readlines()
            last_lines = all_lines[-lines_count:] if len(all_lines) > lines_count else all_lines
        
        return render_template('logs.html', logs=last_lines, count=len(last_lines))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint prediksi
@app.route('/predict', methods=['POST'])
def predict():
    request_type = "unknown"
    start_time = time.time()
    save_debug_sample = True  # Set ke False untuk menonaktifkan penyimpanan sampel
    
    try:
        # Log informasi dasar tentang request
        logger.info(f"Menerima request dari {request.remote_addr} | Method: {request.method} | Content-Type: {request.content_type}")
        
        # Log raw data jika ada
        if request.data:
            data_size = len(request.data)
            data_preview = request.data[:50]
            hex_preview = ' '.join([f'{b:02x}' for b in data_preview])
            logger.info(f"Raw data size: {data_size} bytes | Preview (hex): {hex_preview}...")
        else:
            logger.info("No raw data in request")
        
        # Log request.get_data() jika berbeda dari request.data
        try:
            get_data = request.get_data()
            if get_data != request.data:
                get_data_size = len(get_data)
                get_data_preview = get_data[:50]
                hex_preview = ' '.join([f'{b:02x}' for b in get_data_preview])
                logger.info(f"get_data() size: {get_data_size} bytes | Preview (hex): {hex_preview}...")
        except Exception as e:
            logger.info(f"Error getting request.get_data(): {str(e)}")
        
        # Cek jika request JSON dengan base64
        if request.is_json:
            request_type = "json_file_upload"
            logger.info(f"Menerima request JSON dari {request.remote_addr}")
            
            # Dapatkan data JSON
            try:
                data = request.get_json()
                logger.info(f"JSON keys: {list(data.keys() if isinstance(data, dict) else [])}")
                
                # Cek apakah ada field yang berisi data gambar (file, image, dll)
                image_fields = ['file', 'image', 'data', 'gambar', 'photo', 'picture']
                found_field = None
                image_data = None
                
                for field in image_fields:
                    if field in data and data[field]:
                        found_field = field
                        image_data = data[field]
                        logger.info(f"Found image data in field '{field}'")
                        break
                
                if image_data:
                    try:
                        # Convert data ke bytes jika itu base64
                        
                        # Hilangkan 'data:image/jpeg;base64,' jika ada
                        if isinstance(image_data, str) and ',' in image_data:
                            image_data = image_data.split(',')[1]
                        
                        # Decode base64
                        binary_data = base64.b64decode(image_data)
                        
                        # Simpan ke file sementara
                        temp_path = os.path.join(os.getcwd(), 'temp_upload.jpg')
                        with open(temp_path, 'wb') as f:
                            f.write(binary_data)
                        
                        # Proses gambar
                        img = load_img(temp_path, target_size=(156, 156))
                        img_array = preprocess_image(img)
                        pred = model.predict(img_array)
                        pred_index = np.argmax(pred)
                        confidence = float(pred[0][pred_index])
                        predicted_label = class_names[pred_index]
                        
                        # Simpan sampel untuk debugging jika diaktifkan
                        request_id = str(uuid.uuid4())[:8]
                        if save_debug_sample:
                            sample_path = os.path.join('logs', 'samples', f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{request_id}_{request_type}.jpg")
                            shutil.copy2(temp_path, sample_path)
                            logger.info(f"Sampel gambar disimpan: {sample_path}")
                        
                        # Update sampel dengan hasil prediksi
                        if save_debug_sample:
                            new_sample_path = os.path.join('logs', 'samples', f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{request_id}_{request_type}_{predicted_label}.jpg")
                            shutil.copy2(temp_path, new_sample_path)
                            if os.path.exists(sample_path):
                                os.remove(sample_path)
                            logger.info(f"Sampel gambar dengan prediksi disimpan: {new_sample_path}")
                        
                        # Hapus file sementara
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        result = {
                            'predicted_class': predicted_label,
                            'confidence': format_confidence(confidence),
                            'confidence_percent': f"{format_confidence(confidence)}%",
                            'similarity': get_similarity(confidence)
                        }
                        
                        # Log request berhasil
                        request_id, _ = log_request(request_type, prediction=result)
                        logger.info(f"REQUEST #{request_id} | JSON base64 (field: {found_field}) berhasil diproses | Prediksi: {predicted_label}")
                        
                        return jsonify(result)
                    except Exception as e:
                        log_request(request_type, error=f"Error processing JSON base64: {str(e)}")
                        logger.error(f"Error processing JSON base64: {str(e)}")
                        return jsonify({'error': f"Error processing JSON base64: {str(e)}"}), 400
                else:
                    log_request(request_type, error="No image data found in JSON")
                    return jsonify({'error': 'No image data found in JSON. Use fields like "file", "image", or "data"'}), 400
            except Exception as e:
                log_request(request_type, error=f"Error parsing JSON: {str(e)}")
                logger.error(f"Error parsing JSON: {str(e)}")
                return jsonify({'error': f"Error parsing JSON: {str(e)}"}), 400
        
        # Cek jika request adalah form-urlencoded dengan base64
        elif request.content_type and 'application/x-www-form-urlencoded' in request.content_type:
            request_type = "form_urlencoded_upload"
            logger.info(f"Menerima request form-urlencoded dari {request.remote_addr}")
            
            # Log data yang diterima untuk debugging
            logger.info(f"Form data keys: {list(request.form.keys())}")
            for key in request.form:
                value = request.form[key]
                value_preview = value[:50] + "..." if len(value) > 50 else value
                logger.info(f"Form field: {key} = {value_preview}")
            
            # Log raw data jika form kosong
            raw_data = None
            if not request.form and request.data:
                try:
                    # Coba decode raw data sebagai string
                    raw_data = request.data.decode('utf-8')
                    logger.info(f"Raw data size: {len(raw_data)} bytes")
                    logger.info(f"Raw data preview: {raw_data[:100]}{'...' if len(raw_data) > 100 else ''}")
                    
                    # Coba cari nama file dari raw data (teknik khusus MIT App Inventor)
                    if '&' in raw_data and '=' in raw_data:
                        params = {}
                        for param in raw_data.split('&'):
                            if '=' in param:
                                key, value = param.split('=', 1)
                                params[key] = value
                        
                        logger.info(f"Extracted parameters: {params}")
                        
                        # MIT App Inventor mungkin mengirim nama file sebagai parameter
                        if 'filename' in params:
                            logger.info(f"Found filename parameter: {params['filename']}")
                except:
                    logger.warning("Tidak dapat decode request.data sebagai UTF-8")
            elif not request.form and request.content_length and request.content_length > 0:
                # Coba baca dari buffer input stream untuk permintaan yang kompleks
                try:
                    # MIT App Inventor dapat mengirim data yang tidak selalu dapat diakses melalui request.data
                    # Coba baca langsung dari input stream
                    raw_data_bytes = request.stream.read()
                    logger.info(f"Raw stream data size: {len(raw_data_bytes)} bytes")
                    
                    # Coba decode
                    try:
                        raw_data = raw_data_bytes.decode('utf-8')
                        logger.info(f"Stream data (decoded): {raw_data[:100]}{'...' if len(raw_data) > 100 else ''}")
                    except:
                        logger.info(f"Stream data (hex): {' '.join([f'{b:02x}' for b in raw_data_bytes[:50]])}...")
                    
                    # Simpan data ini untuk diproses
                    if raw_data_bytes:
                        # Simpan untuk pemrosesan nanti
                        temp_raw_path = os.path.join(os.getcwd(), 'temp_stream.jpg')
                        with open(temp_raw_path, 'wb') as f:
                            f.write(raw_data_bytes)
                        logger.info(f"Stream data saved to {temp_raw_path}")
                        
                        # Coba buka sebagai gambar
                        try:
                            from PIL import Image
                            img = Image.open(temp_raw_path)
                            logger.info(f"Stream data successfully opened as image: {img.format} {img.size}")
                            
                            # Jika berhasil, buat FileStorage dan lanjutkan ke pemrosesan
                            from io import BytesIO
                            from werkzeug.datastructures import FileStorage
                            file = FileStorage(
                                stream=BytesIO(raw_data_bytes),
                                filename='stream_data.jpg',
                                content_type='image/jpeg'
                            )
                            
                            # Proses gambar
                            img_array = preprocess_image(img)
                            pred = model.predict(img_array)
                            pred_index = np.argmax(pred)
                            confidence = float(pred[0][pred_index])
                            predicted_label = class_names[pred_index]
                            
                            # Simpan sampel untuk debugging
                            request_id = str(uuid.uuid4())[:8]
                            if save_debug_sample:
                                sample_path = os.path.join('logs', 'samples', f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{request_id}_stream_data.jpg")
                                shutil.copy2(temp_raw_path, sample_path)
                                logger.info(f"Stream data sample saved: {sample_path}")
                            
                            # Hapus file temporary
                            if os.path.exists(temp_raw_path):
                                os.remove(temp_raw_path)
                            
                            result = {
                                'predicted_class': predicted_label,
                                'confidence': format_confidence(confidence),
                                'confidence_percent': f"{format_confidence(confidence)}%",
                                'similarity': get_similarity(confidence)
                            }
                            
                            # Log success
                            request_id, _ = log_request(request_type, prediction=result)
                            logger.info(f"REQUEST #{request_id} | Stream data processed successfully | Prediction: {predicted_label}")
                            
                            return jsonify(result)
                        except Exception as e:
                            logger.error(f"Error processing stream data as image: {str(e)}")
                except Exception as e:
                    logger.error(f"Error reading from request stream: {str(e)}")
            else:
                # Log request headers lebih detail untuk membantu debugging
                logger.info("Request form kosong dan tidak ada raw data")
                logger.info(f"Request headers: {dict(request.headers)}")
                logger.info(f"Content-Length: {request.content_length}")
                logger.info(f"Content-Type: {request.content_type}")
                logger.info(f"User-Agent: {request.user_agent}")
            
            # Jika data kosong atau tidak dapat diproses, berikan panduan penggunaan API
            if not request.form and not raw_data:
                error_message = "Empty request. Untuk MIT App Inventor, gunakan salah satu metode berikut:\n" + \
                               "1. Web.PostFile - URL: http://[server]/predict, parameter Path: path ke file gambar\n" + \
                               "2. Web.PostText - URL: http://[server]/predict, parameter Text: hasil Canvas.toBase64()"
                log_request(request_type, error="Empty form-urlencoded request")
                logger.error(f"Permintaan form-urlencoded kosong dengan detail headers: {dict(request.headers)}")
                return jsonify({'error': error_message}), 400

            # Jika data diterima dalam raw body, coba proses sebagai file gambar
            if request.data:
                try:
                    # Simpan data raw ke file sementara
                    temp_raw_path = os.path.join(os.getcwd(), 'temp_raw.jpg')
                    with open(temp_raw_path, 'wb') as f:
                        f.write(request.data)
                    
                    # Buat FileStorage object untuk mempertahankan konsistensi kode
                    from io import BytesIO
                    from werkzeug.datastructures import FileStorage
                    file = FileStorage(
                        stream=BytesIO(request.data),
                        filename='formdata.jpg',
                        content_type='image/jpeg'
                    )
                    
                    # Simpan file sementara
                    temp_path = os.path.join(os.getcwd(), 'temp_upload.jpg')
                    file.save(temp_path)
                    
                    # Proses gambar dari file yang disimpan
                    img = load_img(temp_path, target_size=(156, 156))
                    img_array = preprocess_image(img)
                    pred = model.predict(img_array)
                    pred_index = np.argmax(pred)
                    confidence = float(pred[0][pred_index])
                    predicted_label = class_names[pred_index]
                    
                    # Simpan sampel untuk debugging jika diaktifkan
                    request_id = str(uuid.uuid4())[:8]
                    if save_debug_sample:
                        sample_path = os.path.join('logs', 'samples', f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{request_id}_{request_type}.jpg")
                        shutil.copy2(temp_path, sample_path)
                        logger.info(f"Sampel gambar disimpan: {sample_path}")
                    
                    # Update sampel dengan hasil prediksi jika diaktifkan
                    if save_debug_sample:
                        new_sample_path = os.path.join('logs', 'samples', f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{request_id}_{request_type}_{predicted_label}.jpg")
                        shutil.copy2(temp_path, new_sample_path)
                        if os.path.exists(sample_path):
                            os.remove(sample_path)
                        logger.info(f"Sampel gambar dengan prediksi disimpan: {new_sample_path}")
                    
                    # Hapus file sementara
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    if os.path.exists(temp_raw_path):
                        os.remove(temp_raw_path)
                    
                    result = {
                        'predicted_class': predicted_label,
                        'confidence': format_confidence(confidence),
                        'confidence_percent': f"{format_confidence(confidence)}%",
                        'similarity': get_similarity(confidence)
                    }
                    
                    # Log request berhasil
                    request_id, _ = log_request(request_type, prediction=result)
                    logger.info(f"REQUEST #{request_id} | Form-urlencoded data berhasil diproses | Prediksi: {predicted_label}")
                    
                    return jsonify(result)
                except Exception as e:
                    log_request(request_type, error=f"Error processing form-urlencoded data: {str(e)}")
                    logger.error(f"Error processing form-urlencoded data: {str(e)}")
                    return jsonify({'error': f"Error processing form-urlencoded data: {str(e)}"}), 400
            
            # Cek jika ada field-field yang umum digunakan
            for field_name in request.form:
                # Coba proses field sebagai data gambar
                try:
                    # Ambil data dari field
                    field_data = request.form[field_name]
                    
                    # Simpan data ke file sementara
                    temp_raw_path = os.path.join(os.getcwd(), 'temp_raw.jpg') 
                    with open(temp_raw_path, 'wb') as f:
                        f.write(field_data.encode('utf-8') if isinstance(field_data, str) else field_data)
                    
                    # Buat FileStorage object
                    from io import BytesIO
                    from werkzeug.datastructures import FileStorage
                    file = FileStorage(
                        stream=BytesIO(field_data.encode('utf-8') if isinstance(field_data, str) else field_data),
                        filename=f'{field_name}.jpg',
                        content_type='image/jpeg'
                    )
                    
                    # Simpan file sementara
                    temp_path = os.path.join(os.getcwd(), 'temp_upload.jpg')
                    file.save(temp_path)
                    
                    # Proses gambar
                    img = load_img(temp_path, target_size=(156, 156))
                    img_array = preprocess_image(img)
                    pred = model.predict(img_array)
                    pred_index = np.argmax(pred)
                    confidence = float(pred[0][pred_index])
                    predicted_label = class_names[pred_index]
                    
                    # Update log dan sampel
                    request_id, _ = log_request(request_type, prediction={'predicted_class': predicted_label, 'confidence_percent': f"{format_confidence(confidence)}%"})
                    logger.info(f"REQUEST #{request_id} | Field '{field_name}' processed | Prediksi: {predicted_label}")
                    
                    # Hapus file sementara
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    if os.path.exists(temp_raw_path):
                        os.remove(temp_raw_path)
                    
                    result = {
                        'predicted_class': predicted_label,
                        'confidence': format_confidence(confidence),
                        'confidence_percent': f"{format_confidence(confidence)}%",
                        'similarity': get_similarity(confidence)
                    }
                    
                    return jsonify(result)
                    
                except Exception as e:
                    logger.warning(f"Field '{field_name}' tidak dapat diproses: {str(e)}")
            
            # Jika sampai di sini, berarti tidak ada field yang dapat diproses
            log_request(request_type, error="Tidak ada field form-urlencoded yang valid")
            return jsonify({'error': 'Tidak ada field yang valid dalam request. Untuk MIT App Inventor, gunakan Web.PostFile untuk mengirim gambar'}), 400
        
        # Cek jika request adalah teks (untuk MIT App Inventor PostText)
        elif request.content_type and 'text/plain' in request.content_type:
            request_type = "text_upload"
            logger.info(f"Menerima request text/plain dari {request.remote_addr}")
            
            if request.data:
                try:
                    # Log preview dari data
                    text_data = request.data.decode('utf-8', errors='replace')
                    text_preview = text_data[:100] + "..." if len(text_data) > 100 else text_data
                    logger.info(f"Text data preview: {text_preview}")
                    logger.info(f"Text data size: {len(request.data)} bytes")
                    
                    # Simpan data raw ke file sementara dan coba buka sebagai gambar
                    temp_raw_path = os.path.join(os.getcwd(), 'temp_raw.jpg')
                    with open(temp_raw_path, 'wb') as f:
                        f.write(request.data)
                    
                    # Buat FileStorage object untuk mempertahankan konsistensi kode
                    from io import BytesIO
                    from werkzeug.datastructures import FileStorage
                    file = FileStorage(
                        stream=BytesIO(request.data),
                        filename='textplain.jpg',
                        content_type='image/jpeg'
                    )
                    
                    # Simpan file sementara
                    temp_path = os.path.join(os.getcwd(), 'temp_upload.jpg')
                    file.save(temp_path)
                    
                    # Proses gambar dari file yang disimpan
                    try:
                        img = load_img(temp_path, target_size=(156, 156))
                        img_array = preprocess_image(img)
                        pred = model.predict(img_array)
                        pred_index = np.argmax(pred)
                        confidence = float(pred[0][pred_index])
                        predicted_label = class_names[pred_index]
                        
                        # Simpan sampel untuk debugging jika diaktifkan
                        request_id = str(uuid.uuid4())[:8]
                        if save_debug_sample:
                            sample_path = os.path.join('logs', 'samples', f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{request_id}_{request_type}.jpg")
                            shutil.copy2(temp_path, sample_path)
                            logger.info(f"Sampel gambar disimpan: {sample_path}")
                        
                        # Update sampel dengan hasil prediksi jika diaktifkan
                        if save_debug_sample:
                            new_sample_path = os.path.join('logs', 'samples', f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{request_id}_{request_type}_{predicted_label}.jpg")
                            shutil.copy2(temp_path, new_sample_path)
                            if os.path.exists(sample_path):
                                os.remove(sample_path)
                            logger.info(f"Sampel gambar dengan prediksi disimpan: {new_sample_path}")
                        
                        # Hapus file sementara
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        if os.path.exists(temp_raw_path):
                            os.remove(temp_raw_path)
                        
                        result = {
                            'predicted_class': predicted_label,
                            'confidence': format_confidence(confidence),
                            'confidence_percent': f"{format_confidence(confidence)}%",
                            'similarity': get_similarity(confidence)
                        }
                        
                        # Log request berhasil
                        request_id, _ = log_request(request_type, prediction=result)
                        logger.info(f"REQUEST #{request_id} | Text/plain data berhasil diproses | Prediksi: {predicted_label}")
                        
                        return jsonify(result)
                    except Exception as e:
                        log_request(request_type, error=f"Error processing text/plain data as image: {str(e)}")
                        logger.error(f"Error processing text/plain data as image: {str(e)}")
                        return jsonify({'error': f"Error processing text/plain data as image: {str(e)}"}), 400
                except Exception as e:
                    log_request(request_type, error=f"Error processing text/plain request: {str(e)}")
                    logger.error(f"Error processing text/plain request: {str(e)}")
                    return jsonify({'error': f"Error processing text/plain request: {str(e)}"}), 400
            else:
                log_request(request_type, error="Empty text/plain request")
                return jsonify({'error': 'Empty text/plain request'}), 400
        
        # Cek apakah ada file di request
        if request.files:
            # Jika file dikirim dengan nama field
            if 'file' in request.files:
                request_type = "form_data_file"
                file = request.files['file']
                logger.info(f"Menerima request form dengan field 'file' dari {request.remote_addr} | Filename: {file.filename}")
            # Jika file dikirim tanpa nama field (App Inventor)
            elif len(request.files) > 0:
                request_type = "form_data_unnamed"
                # Ambil file pertama yang dikirim
                file = list(request.files.values())[0]
                field_name = list(request.files.keys())[0]
                logger.info(f"Menerima request form tanpa nama field dari {request.remote_addr} | Field: '{field_name}' | Filename: {file.filename}")
            else:
                log_request("form_data_empty", error="No file uploaded")
                return jsonify({'error': 'No file uploaded'}), 400
                
            # Log informasi file
            filename = file.filename
            file_size = 0  # Akan diperbarui setelah menyimpan file
                
        # Jika body request langsung berisi file (raw binary)
        elif request.data:
            request_type = "raw_binary"
            data_size = len(request.data)
            logger.info(f"Menerima request raw binary dari {request.remote_addr} | Size: {data_size} bytes")
            # Simpan data raw ke file sementara
            temp_raw_path = os.path.join(os.getcwd(), 'temp_raw.jpg')
            with open(temp_raw_path, 'wb') as f:
                f.write(request.data)
                file_size = len(request.data)
            # Buat FileStorage object untuk mempertahankan konsistensi kode
            from io import BytesIO
            from werkzeug.datastructures import FileStorage
            file = FileStorage(
                stream=BytesIO(request.data),
                filename='file.jpg',
                content_type=request.content_type
            )
            filename = 'raw_binary_data.jpg'
        else:
            log_request("empty", error="No file uploaded")
            return jsonify({'error': 'No file uploaded'}), 400
        
        # Simpan file sementara
        temp_path = os.path.join(os.getcwd(), 'temp_upload.jpg')
        file.save(temp_path)
        
        # Dapatkan ukuran file setelah disimpan
        if 'file_size' not in locals():
            file_size = os.path.getsize(temp_path)
        
        # Proses gambar dari file yang disimpan
        img = load_img(temp_path, target_size=(156, 156))
        image_size = f"{img.width}x{img.height}"
        
        # Simpan sampel untuk debugging jika diaktifkan
        request_id = str(uuid.uuid4())[:8]
        if save_debug_sample:
            # Buat copy dari file untuk sampel
            sample_path = os.path.join('logs', 'samples', f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{request_id}_{request_type}.jpg")
            shutil.copy2(temp_path, sample_path)
            logger.info(f"Sampel gambar disimpan: {sample_path}")
        
        img_array = preprocess_image(img)
        pred = model.predict(img_array)
        pred_index = np.argmax(pred)
        confidence = float(pred[0][pred_index])
        predicted_label = class_names[pred_index]
        
        # Update sampel dengan hasil prediksi jika diaktifkan
        if save_debug_sample:
            new_sample_path = os.path.join('logs', 'samples', f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{request_id}_{request_type}_{predicted_label}.jpg")
            shutil.copy2(temp_path, new_sample_path)
            # Hapus sampel awal tanpa hasil prediksi
            if os.path.exists(sample_path):
                os.remove(sample_path)
            logger.info(f"Sampel gambar dengan prediksi disimpan: {new_sample_path}")
        
        # Hapus file sementara
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if 'temp_raw_path' in locals() and os.path.exists(temp_raw_path):
            os.remove(temp_raw_path)

        

        result = {
            'predicted_class': predicted_label,
            'confidence': format_confidence(confidence),
            'confidence_percent': f"{format_confidence(confidence)}%",
            'similarity': get_similarity(confidence)
        }
        
        # Log request berhasil dengan detail tambahan
        request_id, _ = log_request(request_type, prediction=result)
        
        # Log detail tambahan
        processing_time = time.time() - start_time
        logger.info(f"REQUEST #{request_id} | Filename: {filename} | File size: {file_size} bytes | " +
                   f"Image size: {image_size} | Processing time: {processing_time:.4f}s")
        
        return jsonify(result)

    except Exception as e:
        # Log error
        log_request(request_type, error=str(e))
        logger.error(f"Error saat memproses gambar: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk melihat log detail (dengan autentikasi sederhana)
@app.route('/admin/detailed-logs', methods=['GET'])
def view_detailed_logs():
    # Log akses ke halaman admin
    logger.info(f"Akses ke halaman admin logs dari {request.remote_addr} | Key: {'valid' if request.args.get('key') == 'admin123' else 'invalid'}")
    
    # Autentikasi sederhana dengan parameter query
    admin_key = request.args.get('key', '')
    if admin_key != 'admin123':  # Ganti dengan kunci yang lebih aman di produksi
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Baca file log detail
        log_path = os.path.join(os.getcwd(), 'logs', 'detailed_requests.log')
        if not os.path.exists(log_path):
            return jsonify({'error': 'Detailed log file not found'}), 404
        
        # Ambil jumlah baris yang diminta
        lines_count = min(int(request.args.get('lines', 5000)), 10000)
        
        # Baca file log
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Split berdasarkan pembatas request
        requests = content.split("===== REQUEST ")
        
        # Ambil n request terakhir
        last_requests = requests[-lines_count:] if len(requests) > lines_count else requests[1:]
        
        return render_template('detailed_logs.html', 
                              requests=last_requests, 
                              count=len(last_requests))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint untuk melihat sampel gambar (dengan autentikasi sederhana)
@app.route('/admin/samples', methods=['GET'])
def view_samples():
    # Autentikasi sederhana dengan parameter query
    admin_key = request.args.get('key', '')
    if admin_key != 'admin123':  # Ganti dengan kunci yang lebih aman di produksi
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Baca direktori sampel
        samples_dir = os.path.join('logs', 'samples')
        if not os.path.exists(samples_dir):
            return jsonify({'error': 'Samples directory not found'}), 404
        
        # Dapatkan daftar file sampel
        samples = []
        for filename in os.listdir(samples_dir):
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                file_path = os.path.join(samples_dir, filename)
                file_stat = os.stat(file_path)
                
                # Parse informasi dari nama file
                parts = filename.split('_')
                date_time = parts[0] if len(parts) > 0 else "Unknown"
                request_id = parts[1] if len(parts) > 1 else "Unknown"
                request_type = parts[2].split('.')[0] if len(parts) > 2 else "Unknown"
                predicted_class = parts[3].split('.')[0] if len(parts) > 3 else None
                
                samples.append({
                    'filename': filename,
                    'path': file_path,
                    'size': file_stat.st_size,
                    'date': date_time,
                    'request_id': request_id,
                    'request_type': request_type,
                    'predicted_class': predicted_class
                })
        
        # Urutkan berdasarkan tanggal (terbaru dulu)
        samples.sort(key=lambda x: x['date'], reverse=True)
        
        # Batasi jumlah sampel yang ditampilkan
        max_samples = min(int(request.args.get('max', 50)), 200)
        samples = samples[:max_samples]
        
        return render_template('samples.html', samples=samples, count=len(samples))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mengakses file sampel
@app.route('/admin/samples/<filename>', methods=['GET'])
def get_sample(filename):
    # Autentikasi sederhana dengan parameter query
    admin_key = request.args.get('key', '')
    if admin_key != 'admin123':  # Ganti dengan kunci yang lebih aman di produksi
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Validasi filename untuk keamanan
        if '..' in filename or '/' in filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        # Baca file sampel
        sample_path = os.path.join('logs', 'samples', filename)
        if not os.path.exists(sample_path):
            return jsonify({'error': 'Sample file not found'}), 404
        
        # Kirim file sebagai respons
        from flask import send_file
        return send_file(sample_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint untuk debugging request
@app.route('/debug', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
def debug_request():
    """Endpoint untuk debugging request - mengembalikan semua informasi tentang request yang diterima"""
    try:
        # Kumpulkan informasi tentang request
        request_info = {
            "method": request.method,
            "url": request.url,
            "path": request.path,
            "headers": {k: v for k, v in request.headers.items()},
            "args": {k: v for k, v in request.args.items()},
            "form": {k: v[:50] + "..." if len(v) > 50 else v for k, v in request.form.items()} if request.form else {},
            "files": {k: {
                "filename": v.filename,
                "content_type": v.content_type,
                "size": len(v.read()) if hasattr(v, "read") else "unknown"
            } for k, v in request.files.items()} if request.files else {},
            "is_json": request.is_json,
            "content_type": request.content_type,
            "content_length": request.content_length,
            "remote_addr": request.remote_addr,
            "user_agent": request.headers.get('User-Agent', 'Unknown'),
        }
        
        # Tambahkan JSON body jika ada
        if request.is_json:
            try:
                json_data = request.get_json()
                request_info["json"] = json_data
            except:
                request_info["json"] = "Error parsing JSON"
        
        # Tambahkan data mentah jika ada dan bukan form atau file
        if request.data and not request.form and not request.files:
            try:
                # Coba decode sebagai UTF-8
                request_info["data_text"] = request.data.decode('utf-8')[:200] + "..." if len(request.data) > 200 else request.data.decode('utf-8')
            except:
                # Jika gagal, tampilkan sebagai hex
                request_info["data_hex"] = ' '.join([f'{b:02x}' for b in request.data[:100]]) + "..." if len(request.data) > 100 else ' '.join([f'{b:02x}' for b in request.data])
            
            request_info["data_size"] = len(request.data)
        
        # Log request untuk debugging
        logger.info(f"DEBUG endpoint accessed from {request.remote_addr}")
        
        return jsonify(request_info)
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Jalankan server
if __name__ == '__main__':
    # Buat direktori logs jika belum ada
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Buat direktori samples jika belum ada
    samples_dir = os.path.join('logs', 'samples')
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    
    # Tambahkan file handler untuk menyimpan log ke file
    file_handler = logging.FileHandler('logs/api.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Server API prediksi huruf Arab sedang berjalan di http://127.0.0.1:5000/")
    app.run(debug=True)
