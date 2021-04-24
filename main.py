from flask import *
import constants as c
import test

app = Flask(__name__)


@app.route('/')
def home():
    return jsonify({c.info: "Question Answer ChatBot"})


@app.route('/test', methods=['GET', 'POST'])
def tests():
    return test.execute(request)


if __name__ == '__main__':
    app.run(debug=True)
