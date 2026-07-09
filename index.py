# Import Dependencies
from flask import Flask, render_template, request, redirect, flash, url_for, jsonify
import urllib.request
from werkzeug.utils import secure_filename
from main import getPrediction
from database import init_db, save_prediction, update_feedback, get_analytics
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#################################################
# Flask Setup
#################################################

UPLOAD_FOLDER = 'static/'

app = Flask(__name__)                    
app.secret_key = '8662747133'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure static folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize SQLite database
init_db()

# Route to HTML    
@app.route('/')
def index():
    return render_template('index.html')


@app.route("/", methods = ['POST']) #/file
# Our function for pushing the image to the classifier model
def submit_image():
     if request.method == 'POST':
          if 'file' not in request.files:
               logger.warning('No file part in request')
               flash('No file part')
               return redirect(request.url)
          file = request.files['file']
        # Error message if no file submitted
          if file.filename == '':
            logger.warning('No file selected for uploading')
            flash('No file selected for uploading')
            return redirect(request.url)
        # Return results predictive data
          if file:
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                logger.info(f"Saving uploaded file to {filepath}")
                file.save(filepath)
                
                # Call the model once
                logger.info(f"Requesting prediction for {filename}")
                answer, probability_results, filename = getPrediction(filename)
                
                # Save to database
                conf = 0.0
                try:
                    conf = float(probability_results)
                except:
                    pass # If it's a string like "Golden Retriever" for OOD
                
                prediction_id = save_prediction(filename, answer, conf)
                
                flash(answer)
                flash(probability_results) # accuracy or OOD class name
                flash(filename)
                flash(prediction_id) # pass ID to frontend for feedback loop
            except Exception as e:
                logger.error(f"Error during upload or prediction: {e}")
                flash('Error')
                flash('Something went wrong during prediction.')
                flash('')
                flash('')
                
            return redirect('/')

@app.route('/feedback/<int:prediction_id>', methods=['POST'])
def feedback(prediction_id):
    """API endpoint to receive user feedback (thumbs up/down)."""
    data = request.get_json()
    is_correct = data.get('correct')
    
    if is_correct is not None:
        success = update_feedback(prediction_id, is_correct)
        if success:
            return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route('/dashboard')
def dashboard():
    """Renders the analytics dashboard."""
    stats = get_analytics()
    return render_template('dashboard.html', stats=stats)

if __name__ == "__main__":
    logger.info("Starting Flask Server...")
    app.run(debug=True)
