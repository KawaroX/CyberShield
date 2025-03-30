from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, World!"

@app.route('/api/test')
def test():
    return jsonify({"message": "API is working!"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')  # 监听所有网络接口