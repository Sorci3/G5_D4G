from flask import Flask, request, jsonify, send_from_directory
import webbrowser
import os
from py.main import summarize_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEBSITE_DIR = os.path.join(BASE_DIR, "website")

app = Flask(__name__, static_folder=os.path.join(WEBSITE_DIR, "css"), template_folder=WEBSITE_DIR)

@app.route("/")
def index():
    return send_from_directory(WEBSITE_DIR, "index.html")

@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(os.path.join(WEBSITE_DIR, "css"), filename)

@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(os.path.join(WEBSITE_DIR, "js"), filename)

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")
        optimized = bool(data.get("optimized", False))
        result = summarize_text(text, optimized)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)