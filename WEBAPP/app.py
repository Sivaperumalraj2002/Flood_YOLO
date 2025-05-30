from flask import Flask, render_template, request, url_for, send_from_directory, redirect
from collections import Counter
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from ultralytics.utils.plotting import Annotator
import cv2
import datetime
import g4f

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
REPORT_FOLDER = 'reports'
OUTPUT_FOLDER = 'static/outputs'

# Ensure directories exist
for folder in [UPLOAD_FOLDER, REPORT_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def process_image(results, model, image_path):
    output_path = os.path.join(OUTPUT_FOLDER, 'output_image.jpg')
    image = cv2.imread(image_path)
    for r in results:
        annotator = Annotator(image)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
        img = annotator.result()
        cv2.imwrite(output_path, img)
    return 'outputs/output_image.jpg'

def process_video(video_path, model):
    output_path = os.path.join(OUTPUT_FOLDER, 'output_video.mp4')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    flood_objects = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)
        annotator = Annotator(frame)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                annotator.box_label(b, model.names[int(c)])
                flood_objects.append(model.names[int(c)])

        annotated_frame = annotator.result()
        out.write(annotated_frame)

    cap.release()
    out.release()
    return 'outputs/output_video.mp4', list(set(flood_objects))

def run_object_detection(file_path, is_video=False):
    model_path = r"F:\ABDUL\ABDUL 2024\FLOOD_YOLO\WEBAPP\best.pt"  # Update with your flood detection YOLO model path
    model = YOLO(model_path)

    if is_video:
        output_path, flood_objects = process_video(file_path, model)
        return flood_objects, output_path
    else:
        results = model.predict(file_path)
        flood_counts = Counter(model.names[int(c)] for r in results for c in r.boxes.cls)
        flood_objects = list(flood_counts.keys())
        output_path = process_image(results, model, file_path)
        return flood_objects, output_path

def generate_flood_suggestion(flood_objects):
    prompt = (
        f"You are an expert in flood management and disaster response. Based on the following detected flood-related objects: {', '.join(flood_objects)}, "
        "provide a detailed analysis of the current flood situation, identify potential risks, and suggest immediate actions to mitigate damage and ensure safety. "
        "Provide actionable recommendations and steps to manage the flood effectively."
    )
    try:
        response = g4f.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            top_p=0.9
        )
        suggestion = response.strip() if response else "No suggestion available."
    except Exception as e:
        suggestion = f"Error generating suggestion: {e}"
    return suggestion

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/index')
def upload():
    return render_template('index.html')

@app.route('/process_file', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return render_template('result.html', error="No file part")

    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', error="No selected file")

    allowed_image_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    allowed_video_extensions = {'mp4', 'avi', 'mov'}
    filename = secure_filename(file.filename)
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

    if file_ext not in allowed_image_extensions and file_ext not in allowed_video_extensions:
        return render_template('result.html', error="Invalid file type")

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    is_video = file_ext in allowed_video_extensions
    flood_objects, output_path = run_object_detection(file_path, is_video)
    if not output_path:
        return render_template('result.html', error="Error processing video")

    flood_suggestion = generate_flood_suggestion(flood_objects)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{os.path.splitext(filename)[0]}_{timestamp}_report.txt"
    report_path = os.path.join(REPORT_FOLDER, report_filename)
    with open(report_path, "w") as report_file:
        report_file.write(flood_suggestion)

    return render_template(
        'result.html',
        filename=filename,
        flood_objects=flood_objects,
        output_path=output_path,
        is_video=is_video,
        flood_suggestion=flood_suggestion,
        report_filename=report_filename
    )

@app.route('/static/outputs/<filename>')
def outputs(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/download_report/<filename>')
def download_report(filename):
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)