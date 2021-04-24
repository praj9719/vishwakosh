from flask import *
import constants as c
from ruff import Bot
import test

app = Flask(__name__)

scibot = Bot(c.science)


@app.route('/')
def home():
    return jsonify({c.info: "Question Answer ChatBot"})


@app.route('/bot', methods=['GET', 'POST'])
def bot():
    return scibot.execute(request)


@app.route('/test', methods=['GET', 'POST'])
def tests():
    return test.execute(request)


if __name__ == '__main__':
    app.run(debug=True)
