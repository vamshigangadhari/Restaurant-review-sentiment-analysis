#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import joblib


# In[2]:


data = pd.read_csv('Restaurant_Reviews.tsv',sep='\t')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.isnull().sum()


# In[6]:


data['Liked'].value_counts()


# In[7]:


data.head()


# In[8]:


data['char_count']=data['Review'].apply(len)


# In[9]:


data.head()


# In[10]:


data['word_count']=data['Review'].apply(lambda x :len(str(x).split()))


# In[11]:


data.head()


# In[12]:


import nltk


# In[13]:


nltk.download('punkt')


# In[14]:


data['sent_count']=data['Review'].apply(lambda x : len(nltk.sent_tokenize(str(x))))


# In[15]:


data.head()


# In[16]:


data[data['Liked']==1]['char_count'].mean()


# In[17]:


data[data['Liked']==0]['char_count'].mean()


# In[18]:


import re


# In[22]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[23]:


custom_stopwords = {'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't",
                    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                    'needn', "needn't", 'shan', "shan't", 'no', 'nor', 'not', 'shouldn', "shouldn't",
                    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}

corpus =[]
ps =PorterStemmer()
stop_words = set(stopwords.words("english")) - custom_stopwords

for i in range(len(data)):
    review = re.sub('[^a-zA-Z]',' ',data['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = " ".join(review)
    corpus.append(review)
data['processed_text']=corpus


# In[24]:


data['processed_text']=corpus


# In[25]:


data.head()


# In[26]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=8,background_color="white")


# In[27]:


pos = wc.generate(data[data['Liked']==1]['processed_text'].str.cat(sep=" "))


# In[28]:


plt.imshow(pos)


# In[29]:


negative = wc.generate(data[data['Liked']==0]['processed_text'].str.cat(sep=" "))


# In[30]:


plt.imshow(negative)


# In[31]:


data.head()


# In[32]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[33]:


x=cv.fit_transform(corpus).toarray()


# In[46]:


x.shape
joblib.dump(cv,"count_v_res")


# In[47]:


y=data['Liked']


# In[48]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.20,random_state=42)


# In[49]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rfc =RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred=lr.predict(X_test)
accuracy_score(y_test,y_pred)


# In[50]:


joblib.dump(rfc,'Restaurant_review_model')


# In[ ]:


import tkinter as tk
from tkinter import ttk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import re


class RestaurantReviewApp:
    def __init__(self, master):
        self.master = master
        master.title("Restaurant Review Classification App")
        master.geometry("400x300")  # Set a custom size for the window

        # Load your pre-trained Random Forest model and CountVectorizer
        # Replace 'your_model.pkl' and 'your_vectorizer.pkl' with the actual filenames
        self.model = joblib.load('Restaurant_review_model')
        self.vectorizer = joblib.load('count_v_res')

        # Create and set up widgets
        title_font = ('Helvetica', 16, 'bold')  # Larger font for the title
        self.label = ttk.Label(master, text="Enter your restaurant review:", font=title_font)
        self.label.pack(pady=10)

        self.text_entry = tk.Text(master, height=5, width=40)
        self.text_entry.pack(pady=10)

        # Increase button size and change color on press
        self.classify_button = ttk.Button(master, text="Classify", command=self.classify_review, style='Custom.TButton')
        self.classify_button.pack(pady=10)

        self.result_label = ttk.Label(master, text="")
        self.result_label.pack(pady=10)

        # Style configuration for the button
        self.style = ttk.Style()
        self.style.configure('Custom.TButton', font=('Helvetica', 12), width=15, foreground='black', background='#4CAF50', padding=(10, 5))
        self.style.map('Custom.TButton', foreground=[('pressed', 'black'), ('active', 'white')], background=[('pressed', '!disabled', '#45a049'), ('active', '#4CAF50')])

    def preprocess_text(self, text):
        custom_stopwords = {'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't",
                            'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                            'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                            'needn', "needn't", 'shan', "shan't", 'no', 'nor', 'not', 'shouldn', "shouldn't",
                            'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
        ps = PorterStemmer()
        stop_words = set(stopwords.words("english")) - custom_stopwords

        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if word not in stop_words]
        review = " ".join(review)

        return review    
    def classify_review(self):
        user_input = self.text_entry.get("1.0", "end-1c")
        if user_input:
            processed_input = self.preprocess_text(user_input)
            # Transform the processed_input using the CountVectorizer
            processed_input_vectorized = self.vectorizer.transform([processed_input])
            prediction = self.model.predict(processed_input_vectorized)[0]
            sentiment = "Positive" if prediction == 1 else "Negative"
            self.result_label.config(text=f"Predicted Sentiment: {sentiment}")
        else:
            self.result_label.config(text="Please enter a review before clicking 'Classify'.")

if __name__ == "__main__":
    root = tk.Tk()
    app = RestaurantReviewApp(root)
    root.mainloop()    
    


# In[ ]:




