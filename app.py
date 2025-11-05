from flask import Flask, request, jsonify
import webbrowser
from py.main import summarize_text

app = Flask(__name__, static_folder="website")

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/css/<path:filename>")
def serve_css(filename):
    return app.send_static_file(f"css/{filename}")

@app.route("/js/<path:filename>")
def serve_js(filename):
    return app.send_static_file(f"js/{filename}")

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