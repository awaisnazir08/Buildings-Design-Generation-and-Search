from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['GENERATED_FOLDER'] = 'static/generated'
app.config['MATCHING_FOLDER'] = 'static/matching'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'sketch' not in request.files:
        return redirect(request.url)
    
    file = request.files['sketch']
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Here you would process the image, generate the real sketch and find matching images
        # For demonstration, we'll use placeholder filenames
        generated_image = 'generated_sketch.png'
        matching_images = ['matching1.png', 'matching2.png']

        return render_template('index.html', 
                               original_image=filename,
                               generated_image=generated_image,
                               matching_images=matching_images)
    
    return redirect(request.url)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MATCHING_FOLDER'], exist_ok=True)
    app.run(debug=True)
