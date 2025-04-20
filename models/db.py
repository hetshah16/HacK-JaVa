from pymongo import MongoClient

MONGO_URI = "mongodb+srv://virajsalunke12:giIrcjiGppmlfuoE@cluster0.didsv7j.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)

db = client['streamlit_auth']
users_collection = db['users']
