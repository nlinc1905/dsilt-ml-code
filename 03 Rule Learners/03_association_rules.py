
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

#-------------------------------------------------------------------------------------------------#
#---------------------------------------EDA and Cleaning------------------------------------------#
#-------------------------------------------------------------------------------------------------#

raw_d = pd.read_excel("Online Retail.xlsx", sheet_name="Online Retail")
print(raw_d.head())

def naCol(df, print_result=True):
    """
    Checks for null or missing values in all columns of a Pandas dataframe

    Arguments:
    df: A Pandas dataframe
    print_result: indicates whether or not the output should be printed to the console

    Returns:
    dict: (key:value) = (column_name:number_missing_values)    
    """
    y = dict.fromkeys(df.columns)
    for idx, key in enumerate(y.keys()):
        if df.dtypes[list(y.keys())[idx]] == 'object':
            y[key] = pd.isnull(df[list(y.keys())[idx]]).sum() + (df[list(y.keys())[idx]]=='').sum()
        else:
            y[key] = pd.isnull(df[list(y.keys())[idx]]).sum()
    if print_result:
        print("Number of nulls by column:")
        for k, v in y.items():
            print(k, v)
    return y

naCol(raw_d)
# Documentation says InvoiceNo starting with "C" is cancellation
print(len(raw_d[raw_d['InvoiceNo'].astype('str').apply(lambda x: x[:1])=='C']))
print(set(raw_d['InvoiceNo'].astype('str').apply(lambda x: x[:1])))  # What does "A" mean?
print(raw_d[raw_d['InvoiceNo'].astype('str').apply(lambda x: x[:1])=='A'][:6])

# Are there any transactions with unusual quantities?
print(raw_d.Quantity.value_counts())
print(raw_d[raw_d['Quantity']<0][:6])     # Possibly returns
print(raw_d[raw_d['Quantity']>1000][:6])  # Cheap items in bulk

# Are there any unusual unit prices?
sns.distplot(raw_d['UnitPrice'])
plt.show()
print(raw_d[raw_d['UnitPrice']<0][:6])     # Adjustments
print(raw_d[raw_d['UnitPrice']>1000][:6])  # Postage and Amazon Fees
# Consider removing StockCodes in ['AmazonFee', 'M' , 'Post', 'Dot', B']

# Can a customer have multiple invoices?
print(raw_d[['InvoiceNo', 'CustomerID']].groupby('CustomerID').agg('count'))

# Does an invoice contain multiple items?
items_by_invoice = raw_d[['StockCode', 'InvoiceNo']].groupby('InvoiceNo').agg('count').reset_index()
print(items_by_invoice)
# Are there invoices with only 1 item?  These can be removed.
invoices_to_remove = items_by_invoice[items_by_invoice['StockCode']<2]['InvoiceNo'].tolist()

# Remove data that could produce meaningless or useless rules
clean_d = raw_d[(~raw_d['InvoiceNo'].str.slice(0,1).isin(['C', 'A'])) \
                & (raw_d['Quantity']>0) \
                & (raw_d['UnitPrice']>0) \
                & (~raw_d['StockCode'].isin(['AMAZONFEE', 'M', 'POST', 'DOT', 'B'])) \
                & (~raw_d['InvoiceNo'].isin(invoices_to_remove))] \
                [['InvoiceNo', 'StockCode', 'Description']]

print(clean_d.info())
clean_d['InvoiceNo'] = clean_d['InvoiceNo'].astype('int')
clean_d['Description'] = clean_d['Description'].str.strip()

#clean_d.to_csv('online_retail_clean.csv', index=False)

#-------------------------------------------------------------------------------------------------#
#------------------------------------Apriori Association Rules------------------------------------#
#-------------------------------------------------------------------------------------------------#

trxn_d = clean_d[['InvoiceNo', 'Description']].groupby('InvoiceNo')['Description'].apply(list).reset_index()
te = TransactionEncoder()
te_ary = te.fit(trxn_d['Description']).transform(trxn_d['Description'])
d = pd.DataFrame(te_ary, columns=te.columns_)

# Sense checks
print(len(set(clean_d['Description'])))  #should match nbr rows
print(len(set(clean_d['InvoiceNo'])))    #should match nbr columns
print(clean_d.info())
print(d.info())
print(d.sum()[d.sum()<1])
print(d.sum()[d.sum()==np.inf])
print(d.sum(axis=1)[d.sum(axis=1)<1])
print(d.sum(axis=1)[d.sum(axis=1)==np.inf])

frequent_itemsets = apriori(d, min_support=0.03,
                            use_colnames=False,
                            max_len=None)
print(frequent_itemsets.head())
a_rules = association_rules(frequent_itemsets,
                            metric="confidence",
                            min_threshold=0.7)
print(a_rules.head())
#a_rules.to_csv('online_retail_learned_rules.csv', index=False)
