import os
import json
from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

app = Flask(__name__)


REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests")
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP Request Latency")
RESPONSE_STATUS = Counter(
    "http_response_status_total", "HTTP response status", ["status_code"]
)
CPU_USAGE = Gauge("system_cpu_usage", "CPU Usage Percentage")
RAM_USAGE = Gauge("system_ram_usage", "RAM Usage Percentage")

MODEL_ACCURACY = Gauge("model_accuracy", "Model accuracy score")
MODEL_PRECISION = Gauge("model_precision", "Model precision score")
MODEL_RECALL = Gauge("model_recall", "Model recall score")
MODEL_F1 = Gauge("model_f1_score", "Model F1 score")


def update_model_metrics():
    metrics_path = "../membangun_model/model_metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                MODEL_ACCURACY.set(metrics.get("accuracy", 0))
                MODEL_PRECISION.set(metrics.get("precision", 0))
                MODEL_RECALL.set(metrics.get("recall", 0))
                MODEL_F1.set(metrics.get("f1_score", 0))
        except json.JSONDecodeError:
            print("Gagal membaca model_metrics.json")


@app.route("/metrics", methods=["GET"])
def prometheus_metrics():
    update_model_metrics()
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()

    api_url = "http://127.0.0.1:5005/invocations"  # Endpoint MLflow model serve
    data = request.get_json()

    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(api_url, json=data, headers=headers)
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)

        RESPONSE_STATUS.labels(status_code=response.status_code).inc()
        return jsonify(response.json())

    except Exception as e:
        RESPONSE_STATUS.labels(status_code=500).inc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Prometheus exporter running at http://127.0.0.1:8000/metrics")
    app.run(host="127.0.0.1", port=8000)
