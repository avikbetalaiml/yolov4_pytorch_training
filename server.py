
from flask import Flask, request
from yolov4 import yolov4_training
from zipfile import ZipFile

app = Flask(__name__)


@app.route('/compare_train', methods = ['POST','GET'])

def compare_train():
	values = {}
	epochs = request.args.get('epochs')
	frame_work = request.args.get('frame_work')
	algo = request.args.get('algo')
	project_id = request.args.get('project_id')
	alias_name = request.args.get('alias_name')
	data = request.files['data']
	with ZipFile(data, 'r') as zipObj:
   			# Extract all the contents of zip file in current directory
   			zipObj.extractall('.')
	#if (epochs, frame_work, algo, project_id) not in None:
	if algo =='yolov5' and frame_work == 'pytorch':
		yolov5obj = yolov5_training(project_id=project_id, epochs=epochs, frame_work=frame_work, algo=algo, alias_name = alias_name)
		get_train = yolov5obj.mlflow_init()
		values = {
			"epochs": epochs,
			"frame_work":frame_work,
			"algo":algo,
			"project_id": project_id
		}
		return values
	elif algo =='yolov4' and frame_work == 'pytorch':
		yolov4obj = yolov4_training(project_id=project_id, epochs=epochs, frame_work=frame_work, algo=algo, alias_name = alias_name)
		get_train = yolov4obj.mlflow_init()
		values = {
			"epochs": epochs,
			"frame_work":frame_work,
			"algo":algo,
			"project_id": project_id
		}
		return values
	return {"Status":"Training Failed"}

# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application
	# on the local development server.
	app.run(port = 5004)
