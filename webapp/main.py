from flask import Flask, redirect, request, url_for, render_template, flash
from summary import *

app = Flask(__name__)

@app.route('/', methods=["POST", "GET"])
def hello():
	if request.method=="POST":
		article = request.form.get('article')
		model = request.form.get('model')
		if model=='bert':
			res = bert_model(article)
		else:
			res = t5_model(article)
		return render_template('home.html', summ=res)
	return render_template('home.html')

if __name__ == "__main__":
	app.run(debug=True)
