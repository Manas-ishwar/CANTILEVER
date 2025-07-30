from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from chatbot_response import generate_response  # DialoGPT logic

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.get_json().get('user_input', '').strip()
    print(f"[INFO] Text received: {user_input}")
    
    if not user_input:
        return jsonify({'response': "Please type something."})
    
    bot_response = generate_response(user_input)
    return jsonify({'response': bot_response})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'response': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'response': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print(f"[UPLOAD] File saved to: {filepath}")
    return jsonify({'response': f'File {filename} uploaded successfully!'})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  # Use Render's dynamic port
    print(f"[INFO] Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port)
