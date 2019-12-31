from flask import Flask, send_file

app = Flask(__name__)

from app import routes
