from flask import Flask, request, jsonify
from ml_model import predict_spam

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify_email():
    data = request.json
    email_content = data['email_content']
    prediction = predict_spam(email_content)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
