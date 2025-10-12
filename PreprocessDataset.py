import pandas as pd

# Load dataset
df = pd.read_csv("gender-classifier.csv", encoding='latin1')

#
df = df[['text', 'description', 'gender']]

# Drop rows with missing gender or both text+description empty
df = df.dropna(subset=['gender'])
df = df[df['gender'].isin(['male', 'female'])]


df['text'] = df['text'].fillna('')
df['description'] = df['description'].fillna('')


df['combined_text'] = df['text'] + ' ' + df['description']


df = df[df['combined_text'].str.strip() != '']

# Save combined_text and gender columns
df = df[['combined_text', 'gender']]
df.to_csv("text_gender_dataset.csv", index=False)

print(f"Cleaned dataset saved as text_gender_dataset.csv")
print(f"Total samples: {len(df)}")
print(df.head())
