from flask import Flask, request, jsonify
from flask_cors import CORS
import jwt
import datetime
import pymongo
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'supersecretkey'  # Change this in production

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["auth_db"]
users = db["users"]

# Register endpoint
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    if users.find_one({ "username": data["username"] }):
        return jsonify({ "msg": "User already exists!" }), 400
    
    hashed_pw = generate_password_hash(data["password"])
    users.insert_one({
        "username": data["username"],
        "password": hashed_pw,
        "role": data.get("role", "user")
    })
    return jsonify({ "msg": "User registered successfully!" }), 201

# Login endpoint
@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    user = users.find_one({ "username": data["username"] })

    if not user or not check_password_hash(user["password"], data["password"]):
        return jsonify({ "msg": "Invalid credentials" }), 401

    token = jwt.encode({
        "username": user["username"],
        "role": user["role"],
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }, app.config["SECRET_KEY"], algorithm="HS256")

    return jsonify({ "token": token, "role": user["role"] })

if __name__ == '__main__':
    app.run(debug=True)
