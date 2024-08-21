import torch
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
from .Milvus_db.setup import MilvusManager
from .Inference.generate_image import ImageGenerator
from .Embeddings_model.get_model import EmbeddingsModel
from .Embeddings.generate_embeddings import Embeddings


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'App/static/uploads'
app.config['GENERATED_FOLDER'] = 'App/static/generated'
app.config['MATCHING_FOLDER'] = 'App/static/matching'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Initialize Milvus and ImageGenerator
milvus = MilvusManager()
# milvus.connect()
# milvus.setup_collections()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_generator = ImageGenerator(device)

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

        # Generate the image using the generator model
        generated_tensor = image_generator.generate_image_from_sketch(file_path)

        # Convert the generated image tensor to a PIL Image
        generated_image = torch.squeeze(generated_tensor, 0).cpu()  # Remove batch dimension and move to CPU
        generated_image_pil = Image.fromarray((generated_image.permute(1, 2, 0).numpy() * 255).astype('uint8'))
        # print(generated_image_pil)
        # plt.imshow(generated_image_pil)
        # Save the generated image to the generated folder
        generated_filename = f"generated_{filename}"
        generated_image_path = os.path.join(app.config['GENERATED_FOLDER'], generated_filename)
        generated_image_pil.save(generated_image_path)

        # Find matching images using Milvus (placeholder code)
        matching_images = ['matching1.png', 'matching2.png']  # Replace with actual matching logic

        return render_template('index.html', 
                               original_image=filename,
                               generated_image=generated_filename,
                               matching_images=matching_images)

    return redirect(request.url)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MATCHING_FOLDER'], exist_ok=True)
    app.run(debug=True)
