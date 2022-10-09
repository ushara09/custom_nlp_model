import numpy as np
import pickle

## loading the vectorizer
filename='sentiment_pred_vectorizer'
loaded_vectorizer = pickle.load(open(filename,'rb'))

#vectorizing the input
user_reviews_array=np.array(["his movie made it into one of my top 10 most awful movies. Horrible. <br /><br />There wasn't a continuous minute where there wasn't a fight with one monster or another. There was no chance for any character development, they were too busy running from one sword fight to another. I had no emotional attachment (except to the big bad machine that wanted to destroy them) <br /><br />Scenes were blatantly stolen from other movies, LOTR, Star Wars and Matrix. <br /><br />Examples<br /><br />>The ghost scene at the end was stolen from the final scene of the old Star Wars with Yoda, Obee One and Vader. <br /><br />>The spider machine in the beginning was exactly like Frodo being attacked by the spider in Return of the Kings. (Elijah Wood is the victim in both films) and wait......it hypnotizes (stings) its victim and wraps them up.....uh hello????<br /><br />>And the whole machine vs. humans theme WAS the Matrix..or Terminator.....<br /><br />There are more examples but why waste the time? And will someone tell me what was with the Nazi's?!?! Nazi's???? <br /><br />There was a juvenile story line rushed to a juvenile conclusion. The movie could not decide if it was a children's movie or an adult movie and wasn't much of either. <br /><br />Just awful. A real disappointment to say the least. Save your money."])
user_review_vector = loaded_vectorizer.transform(user_reviews_array)

#loading the model
filename='sentiment_pred_model'
loaded_model = pickle.load(open(filename,'rb'))
loaded_model.predict(user_review_vector)
print(loaded_model.predict(user_review_vector))



