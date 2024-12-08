



































































































import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv(r'F:\suja\Shopping_Trends.csv')
df = df[['Customer ID','Item Purchased','Category']]

transactions = df.groupby('Category')['Item Purchased'].apply(list).values.tolist()

# Convert the dataset to a one-hot encoded format
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Find frequent itemsets with Apriori algorithm
min_support = 0.2
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

# Check if there are any frequent itemsets
if frequent_itemsets.empty:
    print("No frequent itemsets found. Try lowering the min_support threshold.")
else:
    # Generate association rules
    min_confidence = 0.6
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence, num_itemsets=2)

    # Display frequent itemsets
    print("Frequent Itemsets:")
    print(frequent_itemsets)

    # Display association rules
    print("\nAssociation Rules:")
    print(rules)
