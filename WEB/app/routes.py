from app import app
from io import BytesIO
from matplotlib import pyplot as plt
import matplotlib
from flask import send_file, render_template
from .modelload import ModelGenerator
matplotlib.use('agg')

# Instancia del Modelo
model = ModelGenerator()


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/img.png')
def draw_building():
    # Obtener una nueva imagen
    img = model.get_prediction()

    # Crear plot
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(img[0].squeeze())
    ax.axis('off')

    # Enviar datos del plot
    return nocache(fig_response(fig))


# Convertir el plot en bytes
def fig_response(fig):
    img_bytes = BytesIO()
    fig.savefig(img_bytes)
    img_bytes.seek(0)
    return send_file(img_bytes, mimetype='image/png')


# Desabilitar la cache para permitir el refresco de las imagenes
def nocache(response):
    """Add Cache-Control headers to disable caching a response"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response