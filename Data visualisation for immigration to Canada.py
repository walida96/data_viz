#!/usr/bin/env python
# coding: utf-8

# ## Foreigners in Canada

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_can = pd.read_excel('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
                       sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skipfooter=2
                      )
print('Data downloaded and read into a dataframe!')
df_can.head()


# In[3]:


df_can.columns 


# In[4]:


df_can.shape 


# Drop columns that useless for our data analysis goal

# In[5]:


df_can.drop(columns = ['AREA', 'REG', 'DEV', 'Coverage', 'Type'], inplace = True)


# In[6]:


#rename columns to appropriate names
df_can.rename(columns = {'OdName': 'Country', 'AreaName': 'Continent', 'RegName':'Region','DevName': 'Development'}, inplace = True)


# In[7]:


df_can.Development.unique()


# In[8]:


all(isinstance(column, str) for column in df_can.columns)


# In[9]:


df_can.columns = list(map(str, df_can.columns))


# In[10]:


all(isinstance(column, str) for column in df_can.columns)


# In[11]:


#set countries as index for convinient using loc function 
df_can.set_index('Country', inplace = True)


# In[12]:


#add total columns for number of foreigners 
df_can['Total'] = df_can.sum(axis = 1)


# In[13]:


df_can.shape


# In[14]:


#create years list for labels 
years = list(map(str, range(1980,2014)))


# In[15]:


print ('Matplotlib version: ', mpl.__version__)


# In[16]:


###Creating Data visualization 


# In[17]:


df_top5 = df_can.sort_values('Total', ascending = False).head(5)


# In[18]:


df_top5 = df_top5[years].transpose()


# In[19]:


df_top5.plot(kind = 'area',
            stacked=False,
             figsize=(20, 10))
plt.title('Imigration to Canada: Top 5 countries of Imigrants')
plt.xlabel('Years')
plt.ylabel('Number of Imigrants')

plt.show()


# In[20]:


#make another alpha parameter 
df_top5.plot(kind = 'area',
             alpha = 0.25,
             stacked = False,
             figsize = (20, 10))
plt.title('Imigration to Canada: Top 5 countries of Imigrants')
plt.xlabel('Years')
plt.ylabel('Number of Imigrants')

plt.show()


# In[21]:


df_least5 = df_can.sort_values('Total', ascending = False).tail(5)


# In[22]:


df_least5 = df_least5[years].transpose()


# In[23]:


df_least5.plot(kind = 'area',
               alpha = 0.25,
              stacked = False,
              figsize = (20,10))
plt.xlabel('years')
plt.ylabel('number of imigrants')
plt.title('The least numbers of Imigrants to Canada')


# In[24]:


count, bin_edges = np.histogram(df_can['2008'])


# In[25]:


bin_edges


# In[26]:


df_can['2008'].plot(kind = 'hist', figsize = (10,5), xticks = bin_edges)


# In[27]:


#What about Denmark, Norway and Sweden? 
north_countries = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()
north_countries.plot.hist()


# In[28]:


#lets make bins for this plot
count, bin_edges = np.histogram(north_countries, 10)
count


# In[ ]:





# In[29]:


north_countries.plot(kind = 'hist',
                    figsize = (10,5)
                    ,bins = 10
                    ,xticks = bin_edges
                    ,stacked = True
                    ,color = ['azure', 'darkslateblue', 'mediumseagreen'])

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants') 

plt.show()


# In[30]:


df_iceland = df_can.loc['Iceland', years]


# In[31]:


df_iceland.plot(kind = 'bar', figsize = (10,6), rot = 90)
plt.xlabel('years')
plt.ylabel('sum of immigrants')
plt.title('Immigration from Iceland to Canada')
plt.annotate('',                      # s: str. Will leave it blank for no text
             xy=(32, 70),             # place head of the arrow at point (year 2012 , pop 70)
             xytext=(28, 20),         # place base of the arrow at point (year 2008 , pop 20)
             xycoords='data',         # will use the coordinate system of the object being annotated 
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='pink', lw=2)
            )
plt.annotate('2008-2012 Financial Crisis'
            ,xy = (29.8, 60)
            ,rotation = 73
            ,va = 'top'
            ,ha = 'center')


# In[32]:


df_total = df_can['Total']


# In[33]:


mpl.style.use('ggplot')


# In[34]:


df_cont = df_can.groupby('Continent').sum()


# In[35]:


#colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
#explode_list = [0.1, 0, 0, 0, 0.1, 0.1] # ratio for each continent with which to offset each wedge.
#df_cont['Total'].plot(kind = 'pie',
#                     figsize = (10,5),
#                    autopct='%1.1f%%',
#                     startangle = 90,
#                    pctdistance = 1.1,
#                     shadow = True,
#                     explode = explode_list,
#                     labels = None)


# In[36]:


df_japan = df_can.loc['Japan', years]


# In[37]:


df_japan.plot(kind = 'box', figsize = (10,5))
plt.title('Boxplot for Japan immigrants to Canada')
plt.ylabel('number of immigrants')


# To visualize multiple plots together, we can create a figure (overall canvas) and divide it into subplots, each containing a plot. With subplots, we usually work with the artist layer instead of the scripting layer.
# 
# If you prefer to create horizontal box plots, you can pass the vert parameter in the plot function and assign it to False. You can also specify a different color in case you are not a big fan of the default red color.

# In[38]:


figure = plt.figure()
ax1 = figure.add_subplot(1,2,1)
ax2 = figure.add_subplot(1,2,2)
#plot1
df_can.loc['China', years].plot(kind = 'box', color = 'blue', vert = False, ax = ax1)
#plot2
df_can.loc['India', years].plot(kind = 'box', color = 'green', vert = False, ax = ax2)


# In[ ]:





# In[ ]:




