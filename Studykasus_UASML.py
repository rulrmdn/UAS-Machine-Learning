import matplotlib.pyplot as plt
import pandas as pd
import nltk

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from textblob import TextBlob

nltk.download('stopwords')
nltk.download('punkt')

text = """
Long long time ago, in England in Sherwood Forest lived Robin Hood.
When he was a boy, he had been cheated by a few noblemen.
Since then he had decided that he would rob the rich and give what he got to the poor.
The Sheriff of Nottingham had made an advertisement that he would give many rewards for the capture of Robin Hood, nobody had ever caught him.
It was because Robin Hood had a number of friends who served him. They acted as informers.
When the Sheriff had any plan to catch him, they would warn Robin Hood.
Many rich people were scared of going through Sherwood Forest because they knew that Robin Hood would attack them.
The Sheriff couldn’t stand it anymore.
Then he went to ask for the king’s help.
However, the king refused to send any of his men to help in the capture of Robbin Hood.
One day The Sheriff and the noblemen held a competition to choose the best shooter in Nottingham.
It was for capturing Robin Hood.
Robin Hood was an excellent shooter.
Therefore, Robin Hood would participate in the competition to prove that he was the best.
He had been warned by his servant, but Robin wasn’t willing to listen.
The competition began.
William, The Sheriff man, and the man in green were trying for the first prize.
it was time for the last arrow to be shot.
The winner of this round would be declared the best shooter in Nottingham.
William could shot very close to the center.
Then the man in green’s turn made the crowd cheer hysterically.
His arrow went through William’s arrows and the center of the target.
Then he shot two more arrows towards the chair on which the Sheriff sat.
No doubt that the man in green was Robin Hood.
immediately Robin Hood pulled off his black wig and then jumped over a wall onto his waiting horse and was gone.
The Sheriff shouted to his men to catch him, but it was too late.
Robin Hood escaped successfully.
"""

stop_words = set(stopwords.words('english'))
words = word_tokenize(text.lower())
filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

word_freq = Counter(filtered_words)

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

word_freq_df = pd.DataFrame(word_freq.items(), columns=['word', 'frequency']).sort_values(by='frequency', ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(word_freq_df['word'][:10], word_freq_df['frequency'][:10], color='skyblue')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Frequent Words')
plt.show()

blob = TextBlob(text)
sentiment = blob.sentiment

print(f"Sentiment Analysis:\nPolarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")

labels = ['Polarity', 'Subjectivity']
values = [sentiment.polarity, sentiment.subjectivity]

plt.figure(figsize=(6, 4))
plt.bar(labels, values, color=['blue', 'orange'])
plt.title('Sentiment Analysis')
plt.ylim(-1, 1)
plt.show()
