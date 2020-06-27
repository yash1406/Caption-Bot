from flask import Flask,render_template,redirect,request;
import captionit
app = Flask(__name__)

@app.route('/')
def hello():
	return render_template("index.html") 

@app.route('/',methods=['POST'])
def marks():
	if request.method == 'POST':
		f = request.files['userfile']
		path = "./static/{}".format(f.filename)
		f.save(path)
		caption = captionit.caption_this_image(path)
		
		return render_template("index.html",yourcaption=caption) 

if __name__ == '__main__':
	app.run(debug = True,threaded = False)