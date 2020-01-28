#!/usr/bin/env python
# coding: utf-8

# In[1]:


#implementing associatoin rules


# In[2]:


#importing libraries and file


# In[22]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules


# In[23]:


book = pd.read_csv("C:\\Users\\prakruthi\\Desktop\\datascience assignments\\association rules\\book.csv")


# In[24]:


#Apriori Algorithm
frequent_itemsets=apriori(book,min_support=0.005,max_len=3,use_colnames=True)


# In[25]:


book.head()


# In[26]:


#most frequent item sets based on supoort
frequent_itemsets.sort_values('support',ascending=False,inplace=True)


# In[27]:


import matplotlib.pyplot as plt
plt.bar(x=list(range(0,11)),height=frequent_itemsets.support[0:11],color='rgmyk');
plt.xticks(list(range(1,11)),frequent_itemsets.itemset[1:11]);
plt.xlabel('items_sets')
plt.ylabel('support')


# In[28]:


rules=association_rules(frequent_itemsets,metric="lift",min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending=False).head(10)


# In[29]:


# to eliminate redudancy in rules #
def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]


# In[30]:


# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)


# In[31]:


##  Performing algorithm for different support, connfidence value and max length ##

## RULE1 ##
frequent_itemsets1 = apriori(book, min_support=0.007, max_len=4,use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets1.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,11)),height = frequent_itemsets1.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets1.itemsets[1:11]);plt.xlabel('item-sets');plt.ylabel('support')


# In[32]:


rules1 = association_rules(frequent_itemsets1, metric="lift", min_threshold=1)
rules1.head(20)
rules1.sort_values('lift',ascending = False,inplace=True)


# In[33]:


##  RULE2 ##

frequent_itemsets2 = apriori(book, min_support=0.009, max_len=5,use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets2.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,11)),height = frequent_itemsets2.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets2.itemsets[1:11]);plt.xlabel('item-sets');plt.ylabel('support')


rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)
rules2.head(20)
rules2.sort_values('lift',ascending = False,inplace=True)


# In[ ]:


'''
#As min lenth value is changing the rules is changing.
#rules =1054 for support=0.005 and max_len=3
#rules1=4556 for support=0.007 and max_len=4
#rules2=9164 for support=0.009 and max_len=5
'''


# In[ ]:




