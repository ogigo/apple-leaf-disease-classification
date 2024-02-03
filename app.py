from flask import Flask, render_template, request
from inference import predict_image

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):  # Ensure the file has an allowed filename
            # Call predict_image function to get result and solution
            prediction_result = predict_image(file)

            # Pass the prediction_result to the result.html template
            return render_template('result.html', prediction_result=prediction_result)

    # Redirect to the home page if no file is uploaded or an error occurs
    return render_template('index.html', error="Error uploading the file.")

def allowed_file(filename):
    # Add any additional checks for allowed file types if necessary
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

if __name__ == '__main__':
    app.run(debug=True)


