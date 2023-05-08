from flask import Flask,jsonify,request,json
from main import input_predictor

app=Flask(__name__)

displayer=""


@app.route('/data',methods=['GET'])
def index():
    global displayer
    data={"content":displayer}
    return jsonify([data])
    

@app.route('/',methods=['POST'])
def backend():
    global displayer
    request_data=json.loads(request.data)
    print(request_data)
    user_input=request_data['content']
    displayer=input_predictor(user_input)
    return {'201':'todo created successfullly'}


if(__name__=="__main__"):
    app.run(debug=True)