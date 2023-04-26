from flask import Flask
from flask_sqlalchemy import SQLAlchemy

def create_app():
    app = Flask(__name__)
    
    from .views import main
    app.register_blueprint(main)
    
    return app
