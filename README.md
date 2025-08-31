# CANTILEVER – AI Internship Projects

This repository contains the projects completed as part of the **Cantilever AI Internship Program**. The focus of this internship was to apply **Artificial Intelligence (AI)** and **Deep Learning** to build real-world applications with both **Natural Language Processing (NLP)** and **Computer Vision (CV)**.

## Projects

### 1. Chatbot with NLP
- Built a **conversational chatbot** using **Seq2Seq models (LSTM)**.
- **Data preprocessing**: tokenization, embeddings (Word2Vec/GloVe).
- **Model**: Encoder–Decoder architecture for generating responses.
- **Flask Web App**: Simple interface to chat with the bot.
- **Deployment-Ready**: Includes `Procfile` and `render.yaml` for hosting.

Example Interaction:
```
You: hi
Bot: Hello! How can I help you?
```

### 2. Image Captioning
- Implemented an **Encoder–Decoder** model for image captioning.
- **Encoder**: CNN (InceptionV3) extracts image features.
- **Decoder**: LSTM generates natural language captions.
- Trained on benchmark datasets (Flickr8k / MS-COCO).
- **Flask Web App**: Upload an image → get a caption.

Example:
_Input image of a dog running in grass_
_Output caption: “A dog running through a field of green grass.”_

## Tech Stack
- **Languages**: Python
- **Libraries**: PyTorch, TensorFlow, Keras, NLTK, NumPy, Pandas
- **Frameworks**: Flask
- **Deployment**: Heroku / Render
- **Version Control**: Git & Git LFS (for large model files)

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Manas-ishwar/CANTILEVER.git
cd CANTILEVER
```

### 2. Install Dependencies
For chatbot only (minimal):
```bash
pip install -r requirements_chatbot.txt
```

Or full project:
```bash
pip install -r requirements.txt
```

### 3. Run the Chatbot Locally
```bash
python app.py
```
Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 4. Run the Image Captioning App
```bash
python image_captioning_app.py
```

## Deployment
- Heroku Deployment:
  ```bash
  heroku create cantilever-chatbot
  git push heroku main
  heroku open
  ```
- Render Deployment: Uses `render.yaml` configuration.

(Live link can be added here once deployed)

## Repository Structure
```
CANTILEVER/
│── app.py                  # Flask app for chatbot
│── chatbot_response.py     # Chatbot logic
│── training.py             # Training script for chatbot
│── preprocessing.py        # Data preprocessing
│── image_captioning_app.py # Flask app for image captioning
│── requirements.txt        # Full dependencies
│── requirements_chatbot.txt# Minimal chatbot dependencies
│── templates/              # HTML files for web apps
│── model/                  # Saved models (.h5, etc.)
│── Procfile                # For Heroku deployment
│── render.yaml             # For Render deployment
```

## Screenshots
(Add screenshots or GIFs of chatbot interface and image captioning results here for better presentation.)

## Internship Outcomes
- Built and deployed **AI-powered applications** combining NLP and Computer Vision.
- Learned about **end-to-end ML pipelines** (data preprocessing → model training → deployment).
- Gained experience in **Flask web apps** and **cloud deployment**.

Author: Manas Ishwar
