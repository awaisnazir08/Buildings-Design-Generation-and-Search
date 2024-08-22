import torch
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import time
from PIL import Image
from .Milvus_db.setup import MilvusManager
from .Inference.generate_image import ImageGenerator
from .Embeddings_model.get_model import EmbeddingsModel
from .Embeddings.generate_embeddings import Embeddings
from .Search.milvus_search import MilvusSearch, ImagesRetrieval

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'App/static/uploads'
app.config['GENERATED_FOLDER'] = 'App/static/generated'
app.config['MATCHING_FOLDER'] = 'App/static/matching'  # This might not be used if matching images are elsewhere
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

milvus = MilvusManager()

image_generator = ImageGenerator(device)

embeddings_model = EmbeddingsModel(device)
clip_model, preprocess = embeddings_model.load_model()
embeddings_generator = Embeddings(clip_model, preprocess, device)

milvus.connect()
buildings_collection, search_params = milvus.setup_collections()
similarity_search = MilvusSearch(buildings_collection, search_params)

images_retrieval = ImagesRetrieval()

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

        # Generate the image using the generator model
        generated_tensor = image_generator.generate_image_from_sketch(file_path)

        # Convert the generated image tensor to a PIL Image
        generated_image = torch.squeeze(generated_tensor, 0).cpu()  # Remove batch dimension and move to CPU
        generated_image_pil = Image.fromarray((generated_image.permute(1, 2, 0).numpy() * 255).astype('uint8'))
        # Save the generated image to the generated folder
        generated_filename = f"generated_{filename}"
        generated_image_path = os.path.join(app.config['GENERATED_FOLDER'], generated_filename)
        generated_image_pil.save(generated_image_path)
        
        start_time = time.time()
        image_embedding = embeddings_generator.generate_image_embedding(generated_image_path)
        end_time = time.time()
        print(f'Time taken to generate embeddings: {end_time - start_time}')
        
        results = similarity_search.search(image_embedding)
        image_names = images_retrieval.get_matched_images_names(results)
        # matching_images_paths = images_retrieval.get_images_paths(images_names=image_names)

        return render_template('index.html', 
                            original_image=filename,
                            generated_image=generated_filename,
                            matching_images=image_names)

    return redirect(request.url)

# New route to serve matching images
@app.route('/matching_images/<path:filename>')
def matching_images(filename):
    matching_images_dir = 'D:\VS Code Folders\Pix2Pix_Buildings\generated_images_512'
    return send_from_directory(matching_images_dir, filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MATCHING_FOLDER'], exist_ok=True)
    app.run(debug=True)
