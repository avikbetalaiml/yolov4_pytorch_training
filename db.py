import pymysql
from datetime import datetime
class yolo4_db:

	def __init__(self):
		self.host = ''
		self.user= ''
		self.password = ''
		self.database = ''
		self.port = 3306
		self.cur = None
		

	def mysqlconnect(self):
		# To connect MySQL database
		conn = pymysql.connect(
		host=self.host,
		user=self.user, 
		password = self.password,
		db=self.database,
		)

		return conn


	def train_comp(self, project_id=None, epochs = None, frame_work=None, algo=None, run_id=None, alias_name = None):
		connection = self.mysqlconnect()
		cursor = connection.cursor()
		try:
			today = datetime.now()
			date_time = today.strftime("%Y-%m-%d %H:%M:%S")
			#print(prj_nm, input_type)
			# for file in files:
			# 	file.save(file.filename)#'./deep_liveness/output/'+
			# 	file_name = file.filename
			sql = "INSERT INTO train_comp (project_id, algo, frame_work, run_id, alias_name) VALUES (%s, %s, %s, %s, %s)"
			val = (str(project_id), algo, frame_work, run_id, alias_name)
			cursor.execute(sql,val)
			connection.commit()
			return "Success"
		except Exception as e:
			print("Issue in train_comp_method : "+str(e))
			return "Failed"
