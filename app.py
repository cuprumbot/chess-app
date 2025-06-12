from flask import Flask, request, render_template, redirect, url_for
from ultralytics import YOLO
import os
from PIL import Image
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DETECTED_FOLDER'] = 'static'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTED_FOLDER'], exist_ok=True)

# Trained model: yolo8m
model = YOLO("models/trained_model_mid.pt")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # It requires an image if invoked via POST
        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save uploaded image
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Run prediction
            results = model.predict(source=filepath, conf=0.5, save=True, save_txt=False, project='static', name='imgs', exist_ok=True)
            #result_path = os.path.join('results', 'run', os.path.basename(filepath))
            result_path = url_for('static', filename=f'imgs/{os.path.basename(filepath)}')

            # Extract predictions
            predictions = []
            for r in results:
                for b in r.boxes:
                    predictions.append({
                        'label': model.names[int(b.cls)],
                        'confidence': round(float(b.conf), 3)
                    })

            return render_template('result.html', image_path=result_path, predictions=predictions)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
