'''
References: 
- apriori.ipynb Lecture Notes
- Geeks for Geeks article: https://www.geeksforgeeks.org/implementing-apriori-algorithm-in-python/
'''

# Step 1: Import required libraries
import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules 
from mlxtend.preprocessing import TransactionEncoder

# Step 2: Load dataset
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

# Step 3: Preprocessing the data
# Bin continuous variables: Age and BMI into categories
df['age_group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=["18-30", "31-45", "46-60", "61+"])
df['BMI_group'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 40], labels=["Underweight", "Normal", "Overweight", "Obese"])

# Step 4: Select relevant categorical columns for analysis
categorical_columns = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 
                       'PhysActivity', 'Fruits', 'Veggies', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 
                       'Sex', 'Education', 'Income', 'age_group', 'BMI_group']

# Step 5: Create a list to store transactions
transactions = []

# Step 6: Loop through each row in the dataframe and create a list of transactions
#   A transaction will include an item for each "1" value in the selected columns
for _, row in df[categorical_columns].iterrows():
    transactions.append([f'{col}_{row[col]}' for col in categorical_columns if row[col] == 1])

# Step 7: One-hot encoding of the transactions
# The TransactionEncoder will convert the transactions into a binary matrix where '1' = presence and '0' = absence
encoder = TransactionEncoder()
encoded_data = encoder.fit(transactions).transform(transactions)

# Step 8: Convert the encoded data into a DataFrame
df_encoded = pd.DataFrame(encoded_data, columns=encoder.columns_)

# Step 9: Apply the Apriori algorithm to the encoded data
# min_support=0.05 -> looking for itemsets that appear in at least 5% of the transactions
frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)

# Step 10: Generate the association rules from the frequent itemsets
# Use 'lift' as a metric + set a minimum threshold of 1.0 for the lift
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Step 11: Format the output
# Sort the rules by lift and display the top 5 rules
top_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(5)

# Step 12: Display the top 5 rules with proper formatting
for index, row in top_rules.iterrows():
    antecedents = ', '.join(list(row['antecedents']))
    consequents = ', '.join(list(row['consequents']))
    support = row['support']
    confidence = row['confidence']
    lift = row['lift']
    print(f"Antecedents: {antecedents}")
    print(f"Consequents: {consequents}")
    print(f"Support: {support:.4f}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Lift: {lift:.4f}")
    print("-" * 40)  # Adds separator