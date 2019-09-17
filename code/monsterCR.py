import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 



def load_monster(monster_name,db='../csv/monster_compendium.csv'):
	'''
	Search the database of monsters (used in the regression) and return a StatBlock object 
	of the monster, if it exists in the database (exact name needed?)
	'''
	
	pass 


def create_base_monster(monster_name):
	monster = StatBlock(name=monster_name)
	monster.update_stat('HP',33.)
	monster.update_stat('AC',15.)
	monster.update_stat('STR',13.)
	monster.update_stat('DEX',13.)
	monster.update_stat('CON',13.)
	monster.update_stat('WIS',13.)
	monster.update_stat('INT',13.)
	monster.update_stat('CHA',13.)
	monster.update_stat('avg_attack_1',10.)
	monster.update_stat('avg_attack_2',10.)
	monster.update_stat('max_attack_1',10.)
	monster.update_stat('max_attack_2',10.)
	monster.update_stat('Size',1.)
	monster.update_stat('spellcasting',1.)
	return monster 

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
	def __init__(self,name):
		#Store relevant properties as attributes of class
		#Load final dataframe of parameters which will then be initialized for each statblock  
		stat_names = pd.read_csv('../csv/monster_compendium.csv',header=0,nrows=1).columns
		self.stats = pd.DataFrame()
		for i in stat_names:
			setattr(self,i,0.0)
			self.stats[i] = np.array([0.0])
		
		self.name = name
		self.stats['Name'] = np.array([self.name]) 
	
		#Also store the relevant properties as a single-valued dataframe which can be acted on by our models 
		# (and for convenient single variable information transfer)
		


	def update_stat(self,stat,update_value):
		'''
		Convenience function to globally update the quantity of a stat at all places in the StatBlock.
		'''
		self.stats[stat] = update_value 
		setattr(self,stat,update_value)
		 

	def calc_CR(self):
		'''
		For an instantiated StatBlock object, calculate the best-fit CR for the creature 
		given the input parameters, and produce a PDF capturing the spread in CR as predicted by the model
		'''
		model = Model()
		fit = model.predict(self.stats)

	def save_stats(self,db='user_monsters.csv'):
		'''
		Save a created StatBlock to a csv file of user monsters (one is provided by default, 
		but new files can be specified).
		If attempting to save a monster with a name identical to that in the database, the method will ask
		whether to overwrite, or select a new name
		'''

		pass 
	
	def update_HP(self,value):
		self.update_stat('HP',value)
	def update_AC(self,value):
		self.update_stat('AC',value)
	def update_CR(self,value):
		self.update_stat('CR',value)
	def update_STR(self,value):
		self.update_stat('STR',value)
	def update_DEX(self,value):
		self.update_stat('DEX',value)
	def update_CON(self,value):
		self.update_stat('CON',value)
	def update_WIS(self,value):
		self.update_stat('WIS',value)
	def update_INT(self,value):
		self.update_stat('INT',value)
	def update_HP(self,value):
		self.update_stat('HP',value)


class Model(object):
	'''
	Object responsible for executing the regression and/or loading regression weights and 
	which contains methods for calculating stats given other stats. (namely CR given everything else)
	'''
	def __init__(self,design_csv='../csv/monster_compendium.csv'):
		self.design_df_raw = pd.read_csv(design_csv,header=0)
		
	

	def calc_normalizations(self,drop_cols):
		df_vectors = self.design_df_raw.copy().drop(drop_cols,axis=1)
		avg = df_vectors.mean(axis=0)
		return avg


	def statblock_to_structure(self,rich_design_df,drop_cols):
		'''
		Turn full statblock into structure matrix precursor.
		Takes in statblock dataframe
		This function drops name and CR, renormalizes columns, and generates column combinations. 
		Returns structure matrix dataframe
		'''
		design_df = rich_design_df.drop(drop_cols,axis=1)
		norm_constants = self.calc_normalizations(drop_cols)
		design_df = design_df / norm_constants
		design_df_copy = design_df.copy()
		for ci in design_df_copy.columns:
			for cj in design_df_copy.columns:
				if not str(ci+'*'+cj) in design_df.columns:
					design_df[str(ci+'*'+cj)]=design_df_copy[ci]*design_df_copy[cj]

		return design_df 

	def fit(self,stat_fit='CR',log_lam=8.5,plot_fit=False):

		#Specify the DESIGN MATRIX of regress
		drop_cols = ['Name','CHA','Unnamed: 0'] + [stat_fit]

		self.design_df = self.statblock_to_structure(self.design_df_raw,drop_cols)

		self.y = np.array(self.design_df_raw[stat_fit]) / self.design_df_raw[stat_fit].mean()
		self.A = np.hstack( (np.array(self.design_df[i]).reshape(-1,1) for i in self.design_df.columns ) ) 
		self.A = np.nan_to_num(self.A)
		self.C = self.A.T.dot(self.A)
		self.C[np.diag_indices_from(self.C)] += 1.0 / 10 ** log_lam
		self.w = np.linalg.solve(self.C, self.A.T.dot(self.y))
		self.pred_stat = np.dot(self.A,self.w) 
		ndof = self.A.shape[1]
		chi_squared = (self.y-self.pred_stat)**2 / ndof
		print('Chi-squared of fit: {}'.format(np.sum(chi_squared)))
		self.model_weights = self.w 
		if plot_fit:
			fig, ax = plt.subplots()
			ax.plot(self.pred_stat*self.design_df_raw[stat_fit].mean(),self.y*self.design_df_raw[stat_fit].mean(),'.',color='C0',alpha=0.9)
			ax.set_ylabel('{} values from the literature'.format(stat_fit))
			ax.set_xlabel('{} values from monsterCR predictions'.format(stat_fit))
			ax.plot([0,40],[0,40],'k',alpha=0.5,label='1:1 relation')
			ax.legend()
			plt.show()

		
	def predict(self,stat_df,stat):
		'''
		Given a single row-vector stat_df dataframe of a monster, use 
		the calculated model weights to generate a predicted CR 
		'''
		drop_cols=['Name','CHA','Unnamed: 0'] + [stat]
		stat_df = self.statblock_to_structure(stat_df,drop_cols)
		A = np.hstack( (np.array(stat_df[i]).reshape(-1,1) for i in stat_df.columns )  ) 
		if not hasattr(self,'model_weights'):
			self.fit()
		fit_stat = np.dot(A,self.model_weights)*self.design_df_raw[stat].mean()
		print('Predicted {}: {}'.format(stat,fit_stat))
		





