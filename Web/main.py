from flask import Flask, render_template, request
import requests
import json

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        form_value = request.form["query"]
        response = requests.get('http://127.0.0.1:8000/query', headers={"query-params": form_value})
        return render_template("index.html", resp=response.text)
