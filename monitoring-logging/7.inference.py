from flask import Flask, request, jsonify
import joblib
import pandas as pd
from prometheus_client import start_http_server, Summary, Counter, Histogram, Gauge
import time
import os
import psutil  


model_path = os.path.join(os.getcwd(), "..", "workflow-CI", "MLproject", "artifacts", "model", "model.pkl")
model_path = os.path.abspath(model_path)  
model = joblib.load(model_path)

app = Flask(__name__)

REQUEST_COUNT = Counter('inference_requests_total', 'Total number of inference requests')
REQUEST_LATENCY = Histogram('inference_latency_seconds', 'Latency of inference in seconds')
ACCURACY_SCORE = Summary('inference_accuracy_score', 'Accuracy on test set')
PRECISION_SCORE = Summary('inference_precision_score', 'Precision on test set')
RECALL_SCORE = Summary('inference_recall_score', 'Recall on test set')

CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')  # Penggunaan CPU
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')  # Penggunaan RAM
 
start_http_server(8000)

@app.route("/predict", methods=["POST"])
@REQUEST_LATENCY.time()
def predict():
    REQUEST_COUNT.inc()
    try:
        input_data = request.get_json()
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("Running inference server on http://localhost:5000")
    app.run(debug=False, port=5000)
