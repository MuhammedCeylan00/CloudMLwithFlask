from flask import Flask, request, jsonify
from google.cloud import vision_v1p3beta1 as vision
import os
from google.cloud import translate
from flask import Flask, request, jsonify
import os
from google.cloud import speech
import base64
from flask import Flask, request, jsonify
import os
from google.cloud import vision_v1p3beta1 as vision
from flask import Flask, request, send_file
import io
import os
from PIL import Image, ImageDraw
from google.cloud import vision_v1p3beta1 as vision
from flask import Flask, request, jsonify
import os
from google.cloud import translate_v2 as translate
import os
from flask import Flask, request, jsonify
from google.cloud import speech
from flask_cors import CORS



app = Flask(__name__)
CORS(app) 
# Labeling

def tag_objects(image_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/muhammedceylan/desktop/bulutBilisimPython/still-girder-317612-c58250876d2e.json'
    client = vision.ImageAnnotatorClient()

    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.object_localization(image=image)

    tags = [obj.name for obj in response.localized_object_annotations]
    return tags

@app.route('/annotate', methods=['POST'])
def annotate_image():
    image_file = request.files['image']
    image_path = 'temp.jpg'
    image_file.save(image_path)

    tags = tag_objects(image_path)

    response = {
        'tags': tags
    }
    return jsonify(response)


####IMAGE TO TEXT###
@app.route('/extract-text', methods=['POST'])
def extract_text():
    # Gelen resmi al
    image_file = request.files['image']
    image_bytes = image_file.read()

    # Resmi Vision API'ye gönder ve sonucu al
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/muhammedceylan/desktop/bulutBilisimPython/still-girder-317612-c58250876d2e.json'
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)

    # Sonucu işle ve döndür
    texts = []
    for text in response.text_annotations:
        texts.append(text.description)

    return jsonify({'extracted_text': texts})




###FACE DETECTION######
def detect_faces(image_bytes):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/muhammedceylan/desktop/bulutBilisimPython/still-girder-317612-c58250876d2e.json'
    client = vision.ImageAnnotatorClient()

    image = vision.Image(content=image_bytes)
    response = client.face_detection(image=image)

    faces = []
    for face in response.face_annotations:
        faces.append({
            'bounding_box': {
                'vertices': [
                    {'x': vertex.x, 'y': vertex.y} for vertex in face.bounding_poly.vertices
                ]
            },
            'joy_likelihood': face.joy_likelihood,
            'sorrow_likelihood': face.sorrow_likelihood,
            'anger_likelihood': face.anger_likelihood,
            'surprise_likelihood': face.surprise_likelihood,
            'under_exposed_likelihood': face.under_exposed_likelihood,
            'blurred_likelihood': face.blurred_likelihood,
            'headwear_likelihood': face.headwear_likelihood
        })

    return faces

def draw_faces(image_bytes, faces):
    image = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(image)

    for face in faces:
        vertices = face['bounding_box']['vertices']
        draw.polygon([
            (vertices[0]['x'], vertices[0]['y']),
            (vertices[1]['x'], vertices[1]['y']),
            (vertices[2]['x'], vertices[2]['y']),
            (vertices[3]['x'], vertices[3]['y'])
        ], outline='red')

    image_bytes_io = io.BytesIO()
    image.save(image_bytes_io, format='JPEG')
    image_bytes_io.seek(0)

    return image_bytes_io

@app.route('/detect-faces', methods=['POST'])
def detect_faces_endpoint():
    # Gelen resmi al
    image_file = request.files['image']
    image_bytes = image_file.read()

    # Yüzleri tespit et
    detected_faces = detect_faces(image_bytes)

    # Yüzleri çiz ve resmi döndür
    result_image_bytes_io = draw_faces(image_bytes, detected_faces)
    return send_file(result_image_bytes_io, mimetype='image/jpeg')





######TRANSLATE#####
@app.route('/translate', methods=['POST'])
def translate_endpoint():
    data = request.get_json()

    target_language = data['target_language']
    text = data['text']

    translated_text = translate_text(text, target_language)

    return translated_text

def translate_text(text, target_language):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/muhammedceylan/desktop/bulutBilisimPython/still-girder-317612-c58250876d2e.json'
    client = translate.Client()

    result = client.translate(text, target_language=target_language)
    translated_text = result['translatedText']

    return translated_text


####SpeechToText####
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/muhammedceylan/desktop/bulutBilisimPython/still-girder-317612-c58250876d2e.json'

@app.route('/transcribe', methods=['POST'])
def transcribe_endpoint():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_file.save('audio.flac')  # Ses dosyasını kaydet

    transcription = transcribe_speech('audio.flac')

    return jsonify({'transcription': transcription})

def transcribe_speech(audio_file):
    client = speech.SpeechClient()

    with open(audio_file, "rb") as audio_data:
        content = audio_data.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=44100,
        audio_channel_count=2,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    return transcript

if __name__ == '__main__':
    app.run()
