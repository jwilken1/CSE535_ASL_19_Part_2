from flask import Flask
from flask import request
import json

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()
        predictions = {}
        predictions['1'] = 'predicted_buy'
        predictions['2'] = 'predicted_buy'
        predictions['3'] = 'predicted_buy'
        predictions['4'] = 'predicted_buy'
        json_predictions = json.dumps(predictions, indent=1)
        return json_predictions     
    if request.method == 'GET':
        return "You should be posting a model?!"
    return "Flask Server how'd you get here?!"

if __name__ == "__main__":
	app.run(host='localhost', port=5000)