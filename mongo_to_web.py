from config import *
from flask import Flask, render_template, request
import io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import base64

app = Flask(__name__, template_folder="web") 
#mongo connection
collection = mongo_db["image_store"]
# PostgreSQL connection details
conn = postgres_conn

def get_image(image_id):
    document = collection.find_one({"image_id": image_id})
    if document:
        image_data = fs.get(document["image"]) #get metadata
        image = np.frombuffer(image_data.read(), dtype=np.uint8)
        image = np.reshape(image, document["shape"])
        return image
    else:
        return None
    
def show_image(image):
    # if image:
    pil_image = Image.fromarray(image)
    # img = image
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    img_io = io.BytesIO()
    pil_image.save(img_io, 'JPEG', quality=70)

    # Seek to the beginning of the stream
    img_io.seek(0)

    # Encode the image data as base64
    image_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return image_base64
    # else:
    #     return None

# def show_image(image):
#     # if image:
#     img = Image.fromarray(image)
#     # img = image
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     img_plot = plt.imread(img, format='jpg')
#     print("ploting image")
#     return img_plot
#     # else:
#     #     return None

@app.route('/')
def home():
    return render_template('get_image.html')

@app.route('/image', methods=['POST'])
def show_image_from_db():
    image_id = request.form.get('image_id')
    image_data = get_image(image_id)
    img = show_image(image_data)
    return render_template('image.html', image=img)

@app.route('/postgres')
def display_table():
    # Connect to the PostgreSQL database
    results = None
    with conn:
        try:
            with conn.cursor() as cur:
        # Execute a query to fetch data from the table
                cur.execute('SELECT * FROM entrance_log')

                # Fetch all rows from the result set
                results = cur.fetchall()
                conn.commit()
                # Close the database connection
                cur.close()

                # Render the template with the data
        except psycopg2.Error as e:
            conn.rollback()
    return render_template('table.html', rows=results)

if __name__ == '__main__':
    app.run(debug=True)
