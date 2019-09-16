# monsterCR
An interactive tool for Dungeon Masters (in D&amp;D 5e) to quickly accomplish several tasks related to the development of homebrew monsters in their games. 

Namely, monsterCR leverages various compendia of published (and homebrew) D&amp;D monsters to help DM's accomplish several tasks: 

1. For a unique homebrew statblock, determine the "best-fit" CR (challenge rating) of your monster. This is not done by a formula on those inputs, but rather on a regression of all published monsters. 

2. Adapt to CR: Take a desired monster of a certain CR and determine (based on preference) what adjustments to its stats (e.g., HP, AC, +to hit) would change it to a monster of the desired CR. 

3. Party Power Ranking: Given certain detailed inputs about your specific party members (size, level, etc.), determine a probability distribution and power rank score which parametrizes (roughly) the difficulty or deadliness of the encounter. 

The idea behind monsterCR is that enormous effort has been (thankfully) put into determining CR for existing monsters published, and that using all of those statistics to create a multidimensional model for how input statistics map to CR (and other properties) can save DM's time. This work, then, would not be possible without the countless hours that have been spent assembling all of this information into usable spreadsheets. 

# Installation and Usage 
monsterCR is not yet an installable package. The source code can be obtained by cloning this repo, e.g., 
```
git clone https://github.com/prappleizer/monsterCR.git
```
Once you have the cloned directory, you can run monsterCR by opening a python/ipython window in the `code` subdirectory. You can import the classes via, e.g., 
```
import monsterCR as mcr
```
The basic building block of monsterCR is the StatBlock object, which contains all relevant information about a certain creature. You can create a "raw" StatBlock object by calling 
```
my_monster = mcr.StatBlock(name='scary monster')
```
Your `my_monster` instance is spun up with the default stats one might be interested in, which can be accessed either as class attributes, e.g., 
```
print(my_monster.HP)
```
or by viewing the whole statblock in its tabular format, 
```
print(my_monster.stats)
```
Stats can be updated in one of two ways, either by running the `update_stat()` method, 
```
my_monster.update_stat('AC',14)
```
Or by using one of the aliases for individual stats, which simply call the above function, e.g., 
```
my_monster.update_AC(14)
```
You can always view the stats again after to make sure these updates have propogated properly. 

By default, all relevant properties start out at 0.0. There are a few convenience functions implemented to allow you to quickly get up and running with StatBlock objects. 

The first is `load_monster()`, which allows you to load up a monster from the database of published monsters (that monsterCR uses to fit your stats). If you happen to know the exact name of the monster you want to start with, you can call the function via 
```
my_zombie_mod = mcr.load_monster(name='Zombie')
```
# Predicting Challenge Rating 
One main feature of monsterCR is the ability to predict a challenge rating (CR) for your monster given a (roughly) arbitrary set of input stats, relying on a linear regression over a large compendium of several hundred published monsters. There are obvious caveats to such a technique, i.e., special abilities, spells, resistances/immunities, etc., have interwoven affects on CR. Our goal is not to painstakingly parametrize these, but rather simply fit on all of them. You can run a simple fit to the data in the compendium via the built-in method:
```
my_monster.calc_CR()
```
or, if you want to start experimenting more, you can directly create a model object 
```
my_model = mcr.Model()
```
Which has two primary methods, `model.fit()` and `model.predict()`. The first method fits a design matrix of columns (and combinations of columns) of statistics about each monster to the (known) CR, producing a set of weights which describe the relative importance of each column when predicting. This is the true guts of the monsterCR algorithm, and future work will be primarily focused on adding more relevant columns to this design matrix. The second method takes in a single statblock or stat-dataframe and, after pre-processing, predicts a CR given the model weights determined by the ``model.fit()`` method. So, one could fit a monster via 
```
my_monster = mcr.load_monster(name='Zombie')
my_monster.update_AC(20)
model = mcr.Model()
model.fit()
model.predict(my_monster.stats)
```
which would be equivalent to running 
```
my_monster = mcr.load_monster(name='Zombie')
my_monster.update_AC(20)
my_monster.calc_CR()
```
You can read more about inputs to the ``model.fit()`` method, including L2 regularization, in the (soon to come) documentation, if you want to get into adding your own features. 



# API 
