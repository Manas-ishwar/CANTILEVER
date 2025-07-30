from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from chatbot_response import generate_response  # <-- DialoGPT response logic

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.get_json().get('user_input', '')
    print(f"🔹 Text received: {user_input}")
    
    if not user_input.strip():
        return jsonify({'response': "Please type something."})
    
    # Generate reply using DialoGPT
    bot_response = generate_response(user_input)
    return jsonify({'response': bot_response})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'response': ' No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'response': ' No selected file'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print(f"📁 File uploaded: {filepath}")
    return jsonify({'response': f'File {filename} uploaded successfully!'})

if __name__ == "__main__":
    print(" Flask App Started")
    app.run(debug=True)
