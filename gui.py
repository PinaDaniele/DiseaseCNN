from flask import Flask, render_template, request, jsonify
from predict import predicter
import os


app = Flask(__name__, static_folder='./templates/static')
pred = predicter()

@app.route("/")
def homepage():
    return render_template("gui.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error" : "send a file"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error":"filename is empty"}), 400
    
    model_id = request.form["modelId"]
    
    file_path = os.path.join(".\\upload", file.filename)
    file.save(file_path)

    try:
        result = pred.predict(file_path, model_id)
        return {"prediction":result}
    except Exception as e:
        return jsonify({"error":e}), 502



if __name__ == '__main__':
    if not os.path.exists(".\\upload"):
        os.mkdir(".\\upload")
    app.run(debug=True)