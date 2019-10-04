import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import functools
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso


db_use = '../csv/New_great_compendium_v1.csv'

def search_monsters(monster_name,db=db_use):
	'''
	Search the database of monsters (used in the regression) and return a list of monsters
	for which the entered search terms exists in a given name in the compendium. 

	ARGS: 
		monster_name (str): Monster name (or subname) to search. For example, searching 'Orc' will 
		bring up all monsters with 'Orc' somewhere in the name. Case sensitive for now.
	'''
	# Search for all instances of that string within names 
	df = pd.read_csv(db,header=0)
	names = list(df.Name.values)
	subsearch = [i for i in names if monster_name in i]
	print('Search found substring matches with the following:')
	print('----------------------------------------------')
	for i in subsearch:
		print(i)
	print('----------------------------------------------')


def search_by_stat(db=db_use,**kwargs):
	'''
	Find all monsters in the compendium with stats in a certain boundary. 
	Input format is a list of any number of KWARGS with format 
	statname=(low,high) OR statname=bool
	for cases where the column is a "one hot"
	A standard search might be
		search_by_stat(CR=(2,3),AC=(12,14),HP=(50,80))
	or 
		search_by_stat(CR=(2,5),spellcasting=True)
	'''
	df = pd.read_csv(db,header=0)
	query_string = ''
	for i in kwargs.keys():
		if type(kwargs[i]) == bool:
			query_string += '{} == 1.0'.format(i)
		else:
			query_string +='{0} > {1} & {0} < {2}'.format(i,kwargs[i][0],kwargs[i][1])
		query_string +=' & '
	query_string = query_string[:-3]
	return df.query(query_string)


def load_monster(monster_name,fullsearch=False,db=db_use):
	'''
	Given a monster name that is in the compendium, create a StatBlock object with that monster's stats. 
	Has an optional fullsearch flag which reproduces the usage of 'search_monsters()'. 

	ARGS: 
		monster_name (str): Name of monster to load (case sensitive for now)
		fullsearch (optional): True or False (default False). If True, searches instead.
	'''
	df = pd.read_csv(db,header=0)
	
	if fullsearch==False:
		#Search for exact string
		search_result = df[df['Name']==monster_name] 
		if search_result.empty:
			print('Monster not found! Try Search...')
			return None
		result_statblock = StatBlock(name=monster_name) 
		result_statblock.update_from_monster(search_result)
		return result_statblock

	elif fullsearch==True: 
		# Search for all instances of that string within names 
		names = list(df.Name.values)
		subsearch = [i for i in names if monster_name in i]
		print('Search found substring matches with the following:')
		print('----------------------------------------------')
		for i in subsearch:
			print(i)
		print('----------------------------------------------')

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
	def __init__(self,name,db=db_use):
		#Store relevant properties as attributes of class
		#Load final dataframe of parameters which will then be initialized for each statblock  
		stat_names = pd.read_csv(db,header=0,nrows=1).columns
		self.all_monsters = pd.read_csv(db,header=0)
		self.stats = pd.DataFrame()

		for i in stat_names:
			setattr(self,i,0.0)
			self.stats[i] = np.array([0.0])
		
		self.name = name
		self.stats['Name'] = np.array([self.name]) 
	
		#Also store the relevant properties as a single-valued dataframe which can be acted on by our models 
		# (and for convenient single variable information transfer)
		

	def update_from_monster(self,df,db=db_use):
		stat_names = pd.read_csv(db,header=0,nrows=1).columns
		for i in stat_names:
			setattr(self,i,df[i].values[0])
			self.stats[i] = np.array(df[i])

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
	
	def radar_plot(self,cols_show=['HP','AC','STR','DEX','CON','avg_attack_1'],normalize=True,just_in_CR=False):
		'''
		Diagnostic plot which shows a 'radar plot' of desired statistics for your monster.
			INPUTS: 
					cols_show (arr_like): list of strings containing stats to show in plot
					normalize (bool): whether to normalize stats comparing to other monsters (default: True)
					just_in_CR (bool): whether to normalize just to creatures at your monsters CR, or to 
									   all creatures in the list. Only used if normalize is True. (default: False)
		    RETURNS: 
		    	fig: A figure showing the radar plot of your creature. If normalize is True, a black circle
		    	     is plotted at 1.0 to demarcate the average. 
		'''

		labels=np.array(cols_show)
		stats=self.stats.loc[0,labels].values
		if normalize:
			if just_in_CR:
				stats_norm = stats / self.all_monsters.loc[self.all_monsters['CR']==self.CR,labels].mean(axis=0)
			else:
				stats_norm = stats / self.all_monsters[labels].mean(axis=0)
		angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
		# close the plot
		stats=np.concatenate((stats,[stats[0]]))
		stats_norm = np.concatenate((stats_norm,[stats_norm[0]]))
		angles=np.concatenate((angles,[angles[0]]))
		fig=plt.figure(figsize=(6,6))
		ax = fig.add_subplot(111, polar=True)
		if normalize:
			ax.plot(angles, stats_norm, 'o-', linewidth=2)
			ax.fill(angles, stats_norm, alpha=0.25)
			ax.set_thetagrids(angles * 180/np.pi, labels)
		else:
			ax.plot(angles, stats, 'o-', linewidth=2)
			ax.fill(angles, stats, alpha=0.25)
			ax.set_thetagrids(angles * 180/np.pi, labels)

		title = self.Name + " | CR {}".format(self.CR)
		ax.set_title(title)
		#ax.set_rticks([0.25,0.5,0.75,1, 1.5, 2]) 
		ax.set_rlabel_position(-25.5)
		rlabels = ax.get_ymajorticklabels()
		for label in rlabels:
			label.set_color('r')
		if normalize:
			rs = np.ones(1000)
			theta = np.linspace(0,2 * np.pi,1000)
			ax.plot(theta, rs,'k',lw=2)
		ax.grid(True)
		return fig
	
	def input_damage_roll(self,damage_string,default=1):
		'''
		Given a typical string version of a dnd damage roll (e.g., 10d5+4),
		calculate the average damage of that roll and the max, and update the relevant stats. 
		If one is provided alone or in a list, it goes into attack_1, unless specified 
		If two strings are provided in a list ['1d5+4','2d6-1'], by default it will fill attack_1 and attack_2 stats. 
		If more than 2 are entered, by default, all stats following the first are lumped into attack_2. (we only track 2 attacks)
		'''
		#Check input 
		if type(damage_string) == 'str':
			damage_string = self.split_damage_rolls(damage_string)
		

		if len(damage_string) == 1:
			avg_attack = self.calc_avg_damage(damage_string[0])
			max_attack = self.calc_max_damage(damage_string[0])
			self.update_stat('avg_attack_1',avg_attack)
			self.update_stat('max_attack_1',max_attack)
			self.update_stat('avg_attack_dmg',avg_attack)
			self.update_stat('max_attack_dmg',max_attack)
		elif len(damage_string) == 2:
			avg_attack_1 = self.calc_avg_damage(damage_string[0])
			avg_attack_2 = self.calc_avg_damage(damage_string[1])
			max_attack_1 = self.calc_max_damage(damage_string[0])
			max_attack_2 = self.calc_max_damage(damage_string[1])
			self.update_stat('avg_attack_1',avg_attack_1)
			self.update_stat('max_attack_1',max_attack_1)
			self.update_stat('avg_attack_2',avg_attack_2)
			self.update_stat('max_attack_2',max_attack_2)
			self.update_stat('avg_attack_dmg',np.sum([avg_attack_1,avg_attack_2]))
			self.update_stat('max_attack_dmg',np.sum([max_attack_1,max_attack_2]))
		elif len(damage_string)>2:
			avg_attack_1 = self.calc_avg_damage(damage_string[0])
			max_attack_1 = self.calc_max_damage(damage_string[0])
			self.update_stat('avg_attack_1',avg_attack_1)
			self.update_stat('max_attack_1',max_attack_1)
			other_avg_attacks = np.sum([self.calc_avg_damage(i) for i in damage_string[1:]]) 
			other_max_attacks = np.sum([self.calc_max_damage(i) for i in damage_string[1:]])
			self.update_stat('avg_attack_2',other_avg_attacks)
			self.update_stat('max_attack_2',other_max_attacks)
			self.update_stat('avg_attack_dmg',np,sum([other_avg_attacks,avg_attack_1]))
			self.update_stat('max_attack_dmg',np,sum([other_max_attacks,max_attack_1]))
	    	


	def split_damage_rolls(self,damage_string):
		'''
		Given a damage_string, determine if more than one are present and if so, split them. 
		e.g., input --> 1d4 returns 1d4, but input --> 1d4+2d6 returns ['1d4',2d6']. 
		Function also catches modifiers, e.g., input--> 6d12+5+1d6-2 returns ['6d12+5','1d6-2']
		'''
		d_locs = [m.start() for m in re.finditer('d', damage_string)]
		d2 = d_locs[-1]
		plus_locs= [m.start() for m in re.finditer('\+', damage_string)]
		diffs = d2 - np.array(plus_locs)
		corr_ind = d2 - np.min([n for n in diffs  if n>0])
		return [damage_string[:corr_ind],damage_string[corr_ind+1:]]
    
	def calc_max_damage(self,damage_string):
		'''
		Given a string in the d&d format 1d8+5 (etc.) or supplemental format 1d8+4+1d6+2,
		return the maximum damage for that roll.
		'''
		if damage_string == 'nan' or damage_string=='VARIES':
			 return 0.0
		else:
			mult = damage_string.replace('d','*')
			max_dmg = eval(mult)
			return max_dmg
	    
	    
	def calc_avg_damage(self,damage_string,combine=True):
		'''
		Given a string in the d&d format 1d8+5 (etc.) or supplemental format 1d8+4+1d6+2,
		return the average damage for that roll.
		'''
		if damage_string == 'nan' or damage_string=='VARIES' or damage_string == 'NaN':
			return 0.0
		else:
			num_roll = damage_string.count('d')
			if num_roll > 1:
				rolls = split_damage_rolls(damage_string)
			else:
				rolls = [damage_string]

			avg_damages = []
			for dmg_str in rolls:
				pos_d = dmg_str.find('d')
				if pos_d == -1:
					return 0.0
				num_dice = int(dmg_str[:pos_d])
				mult = dmg_str.replace('d','*0.5*')
				avg_dmg = eval(mult) 
				avg_damages.append(avg_dmg + 0.5*num_dice)

			if combine:
				return np.sum(avg_damages)
			elif combine==False:
				return avg_damages

class Model(object):
	'''
	Object responsible for executing the regression and/or loading regression weights and 
	which contains methods for calculating stats given other stats. (namely CR given everything else)
	'''
	def __init__(self,design_csv=db_use):
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

	def fit(self,stat_fit='CR',regression_type='sklearn_linear',log_lam=8.5,plot_fit=False):

		#Specify the DESIGN MATRIX of regress
		drop_cols = ['Name'] + [stat_fit]
		if regression_type == 'matrix':
			self.design_df = self.statblock_to_structure(self.design_df_raw,drop_cols)
		elif regression_type == 'sklearn_linear':
			self.design_df = self.design_df_raw.drop(['Name',stat_fit],axis=1)
		self.y = np.array(self.design_df_raw[stat_fit]) #/ self.design_df_raw[stat_fit].mean()
		if regression_type=='matrix':
			self.A = np.hstack( (np.array(self.design_df[i]).reshape(-1,1) for i in self.design_df.columns ) ) 
			self.A = np.nan_to_num(self.A)
			self.C = self.A.T.dot(self.A)
			self.C[np.diag_indices_from(self.C)] += 1.0 / 10 ** log_lam
			self.w = np.linalg.solve(self.C, self.A.T.dot(self.y))
			self.pred_stats = np.dot(self.A,self.w) 
			ndof = self.A.shape[1]
			chi_squared = (self.y-self.pred_stat)**2 / ndof
			print('Chi-squared of fit: {}'.format(np.sum(chi_squared)))
			self.model_weights = self.w 
		
		elif regression_type=='sklearn_linear':
			self.model = RidgeCV(alphas=np.logspace(-10,4, 6),normalize=True)
			#self.model = LinearRegression()
			self.model.fit(self.design_df,self.y)
			self._isfit = True
			self.pred_stats = np.rint(self.model.predict(self.design_df))
			print(self.model.score(self.design_df,self.y))
		if plot_fit:
			fig, ax = plt.subplots()
			ax.plot(self.pred_stats,self.y,'.',color='C0',alpha=0.9)
			ax.set_ylabel('{} values from the literature'.format(stat_fit))
			ax.set_xlabel('{} values from monsterCR predictions'.format(stat_fit))
			ax.plot([0,33],[0,33],'k',alpha=0.5,label='1:1 relation')
			ax.legend()
			plt.show()

		
	def predict(self,stat_df,stat,mode='sklearn'):
		'''
		Given a single row-vector stat_df dataframe of a monster, use 
		the calculated model weights to generate a predicted CR 
		'''
		drop_cols=['Name'] + [stat]
		
		if not hasattr(self,'_isfit'):
			self.fit()
		if mode=='matrix':
			stat_df = self.statblock_to_structure(stat_df,drop_cols)
			A = np.hstack( (np.array(stat_df[i]).reshape(-1,1) for i in stat_df.columns )  ) 
			fit_stat = np.dot(A,self.model_weights)*self.design_df_raw[stat].mean()
		elif mode =='sklearn':
			A = stat_df.drop(['Name',stat],axis=1)
			fit_stat = self.model.predict(A)
			if fit_stat < 0.0:
				fit_stat = 0.0
		print('Predicted {}: {}'.format(stat,fit_stat))
		





