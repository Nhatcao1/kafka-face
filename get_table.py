import sys
sys.path.append('/Users/nhatcao/multi-topics-video-stream')
from config import *

from flask import Flask, render_template

app = Flask(__name__, template_folder='web')

# PostgreSQL connection details
conn = postgres_conn
# Route to display the table
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