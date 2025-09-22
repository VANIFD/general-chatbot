from flask import Flask, render_template, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os

# 🔑 Read Google API Key from file
with open("API_key.txt", "r") as f:
    GOOGLE_API_KEY = f.read().strip()

# 🧠 Initialize LangChain LLM with memory
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    convert_system_message_to_human=True,
)

# 🧠 Memory to store conversation
memory = ConversationBufferMemory()

# 🔗 Setup LangChain conversation chain
conversation = ConversationChain(llm=llm, memory=memory)

# ✅ FIXED HERE: __name__ (not _name_)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_response():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    try:
        response = conversation.run(user_input)
        print("LangChain reply:", response)
        return jsonify({"reply": response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ FIXED HERE: __name__ and '__main__'
if __name__ == '__main__':
    app.run(debug=True)
