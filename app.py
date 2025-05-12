from flask import Flask,jsonify,request
from transformers import pipeline
from flask_cors import CORS
app=Flask(__name__)
CORS(app)
# Load the detector pipeline
detector = pipeline("text-classification", model="roberta-base-openai-detector")

@app.route("/test",methods=["post"])
def detect_text():
    try:
        data=request.json
        text=data["text"]
        if text is None:
            return jsonify({"error":"No text provided"})
        result=detector(text)
        label=result[0]['label']
        score=result[0]['score']
        return jsonify({"label":label,"score":score})
    except Exception as e:
        return jsonify({"error":str(e)})
    
if __name__=="__main__":
    app.run(debug=True)