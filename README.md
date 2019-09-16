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
my_monster.updata_AC(14)
```
You can always view the stats again after to make sure these updates have propogated properly. 



# API 
