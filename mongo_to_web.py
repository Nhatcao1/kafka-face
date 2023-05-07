from flask import Flask, render_template
from pymongo import MongoClient

app = Flask(__name__, template_folder='template')
client = MongoClient('mongodb://localhost:27017')
db = client['test']
collection = db['Employee']

@app.route('/')
def index():
    data = collection.find()
    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)