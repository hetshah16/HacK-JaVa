import bcrypt
import jwt
import datetime
from db import users_collection

JWT_SECRET = 'your_jwt_secret'

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def register_user(username, password, role="user"):
    if users_collection.find_one({"username": username}):
        return False
    hashed = hash_password(password)
    users_collection.insert_one({
        "username": username,
        "password": hashed,
        "role": role
    })
    return True

def login_user(username, password):
    user = users_collection.find_one({"username": username})
    if not user or not verify_password(password, user["password"]):
        return None
    payload = {
        "username": user["username"],
        "role": user["role"],
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
    return token

def decode_token(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        return None
