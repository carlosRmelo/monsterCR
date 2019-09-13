import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 



def load_monster(monster_name,db='../csv/monster_compendium.csv'):
	'''
	Search the database of monsters (used in the regression) and return a StatBlock object 
	of the monster, if it exists in the database (exact name needed?)
	'''
	
	pass 


def view_user_monsters(db='../csv/user_monsters.csv'):
	'''
	Load up the csv (to df) containing user-defined monsters stored on-file and view it 
	'''
	pass 


class StatBlock(object):
	'''
	The primary user object in monsterCR -- the StatBlock contains all relevant statistics on a creature
	along with methods for calculating (or recalculating) desired statistics
	'''
	def __init__(self,name,creature_type='monster'):
		#Store relevant properties as attributes of class
		#Load final dataframe of parameters which will then be initialized for each statblock  
		stat_names = pd.read_csv('../csv/monster_stats_with_attack_rolls.csv',header=0,nrows=1).columns
		self.stat_df = pd.DataFrame()
		for i in stat_names:
			setattr(self,i,0.0)
			self.stat_df[i] = np.array([0.0])
		
		self.name = name
		self.stat_df['name'] = np.array([self.name]) 
		self.creature_type = creature_type 
		self.stat_df['creature_type'] = np.array([creature_type])
		#Also store the relevant properties as a single-valued dataframe which can be acted on by our models 
		# (and for convenient single variable information transfer)
		


	def update_stat(self,stat,update_value):
		'''
		Convenience function to globally update the quantity of a stat at all places in the StatBlock.
		'''
		self.stat_df[stat] = update_value 
		setattr(self,stat,update_value)
		 

	def calc_CR(self):
		'''
		For an instantiated StatBlock object, calculate the best-fit CR for the creature 
		given the input parameters, and produce a PDF capturing the spread in CR as predicted by the model
		'''
		model = Model()
		fit = model.predict(self.stat_df)

	def save_stats(self,db='user_monsters.csv'):
		'''
		Save a created StatBlock to a csv file of user monsters (one is provided by default, 
		but new files can be specified).
		If attempting to save a monster with a name identical to that in the database, the method will ask
		whether to overwrite, or select a new name
		'''

		pass 


class Model(object):
	'''
	Object responsible for executing the regression and/or loading regression weights and 
	which contains methods for calculating stats given other stats. (namely CR given everything else)
	'''
	def __init__(self,design_csv):
		self.design_df = pd.read_csv(design_csv,header=0)

	def fit(self):
		
		#Specify the DESIGN MATRIX of regress
		
		y = design_df['CR']
		A_prec = self.statblock_to_structure(self.design_df)
		A = np.hstack( (np.array(design_df[i]).reshape(-1,1) for i in design_df.columns ) ) 
		C = A.T.dot(A)
		w = np.linalg.solve(C, A.T.dot(y))
		pred_cr = np.dot(A,w)
		ndof = A.shape[1]
		chi_squared = (y-pred_cr)**2 / ndof
		self.model_weights = w 
		fig, ax = plt.subplots()
		ax.plot(pred_cr,y,color='C0','.',alpha=0.4)
		plt.show()

		
	def predict(self,stat_df):
		'''
		Given a single row-vector stat_df dataframe of a monster, use 
		the calculated model weights to generate a predicted CR 
		'''
		A = np.hstack( (np.array(stat_df[i]).reshape(-1,1) for i in stat_df.columns )  ) 
		if not hasattr(self,'model_weights'):
			self.fit()
		fit_cr = np.dot(A,self.model_weights)
		print('Predicted CR: {}'.format(pred_cr))
		self.fit_CR = fit_cr
		




