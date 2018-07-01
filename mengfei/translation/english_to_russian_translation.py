from googletrans import Translator
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm



train_df = pd.read_csv("../input/train.csv", parse_dates = ["activation_date"])
train_df = shuffle(train_df, random_state=1234); train_df = train_df.iloc[:100000]
train_df["title"].fillna("nicapotato",inplace=True)
train_df["description"].fillna("nicapotato",inplace=True)

translator = Translator(
            service_urls=[
            'translate.google.de',
            'translate.google.com',
            'translate.google.ru',            
            ]
            )
translator = Translator()
result = []
result2 = []

for index in tqdm(range(0, len(train_df.title))):      
      try:   
            translations = translator.translate(train_df.iloc[index]["title"].lower(), src='en', dest='ru')
            translations2 = translator.translate(train_df.iloc[index]["description"].lower(), src='en', dest='ru')  
            result.append(translations.text)
            result2.append(translations2.text)
      except:           
            result.append(train_df.iloc[index]["title"].lower())
            result2.append(train_df.iloc[index]["description"].lower())
print(result)

train_df['title_2'] = result
train_df['description_2'] = result2

train_df.to_csv("train_trains_100k.csv", index=False)