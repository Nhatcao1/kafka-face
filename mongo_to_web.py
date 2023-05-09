from config import *
from flask import Flask, render_template, request
import io
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder="web") 
#mongo connection
collection = mongo_db["image_store"]
# PostgreSQL connection details
conn = postgres_conn

def get_image(image_id):
    document = collection.find_one({"image_id": image_id})
    if document:
        image_data = document["image"]
        return image_data
    else:
        return None

def show_image(image_data):
    if image_data:
        image_bytes = io.BytesIO(image_data)
        img = plt.imread(image_bytes, format='jpg')
        return img
    else:
        return None

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



####
# db.createCollection("image_store", {
#   validator: {
#     $jsonSchema: {
#       bsonType: "object",
#       required: ["image_id", "image"],
#       properties: {
#         image_id: {
#           bsonType: "string"
#         },
#         image: {
#           bsonType: "binData",
#           description: "Image data in JPG format"
#         }
#       }
#     }
#   }
# })

# def get_image(image_id):
#     document = collection.find_one({"image_id": image_id})
#     if document:
#         image_data = document["image"]
#         return image_data
#     else:
#         return None

# def show_image(image_data):
#     if image_data:
#         image_bytes = io.BytesIO(image_data)
#         img = plt.imread(image_bytes, format='jpg')
#         plt.imshow(img)
#         plt.axis('off')
#         plt.show()
#     else:
#         print("Image not found")

# # Usage example
# image_id = "your_image_id"
# image_data = get_image(image_id)
# show_image(image_data)

# # Close the MongoDB connection
# client.close()