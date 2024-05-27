from flask import Flask, render_template, request, jsonify 
from backend_program import search

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_strings():
    string1 = request.form['videoUrl'] 
    string2 = request.form['query'] 
    output_list = search(string1, string2)
    return jsonify({'output': output_list}) 

if __name__ == '__main__':
    app.run(debug=True)






