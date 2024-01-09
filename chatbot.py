import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download("punkt")
import json
import pickle


import numpy as np
import random
from keras.models import Sequential
from keras.layers import Activation,Dropout,Dense
from keras.optimizers import SGD
from keras.models import load_model

#chatbot gui
from termcolor import colored
words=[]
documents=[]
classes=[]
ignore_words=['?','!','.','“', '”','(', ')', ',','’']

data_file = open('C:/Users/Administrator/Downloads/flask/intents.json')
intents=json.load(data_file)
nltk.download('wordnet')

for intent in intents['intents']:
  for pattern in intent['patterns']:
    w=nltk.word_tokenize(pattern)
    words.extend(w)

    documents.append((w,intent['tag']))
    # add to our classes list
    if intent['tag'] not in classes:
      classes.append(intent['tag'])

#lemizer and lower each word and remove duplicates
words=[lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words=sorted(list(set(words)))

#sort classes
classes=sorted(list(set(classes)))

#decuments= pattern , intents
print(len(documents),"documents")
#classes= intents
print(len(classes),"classes",classes)
#words = each words , vocabulary
print(len(words),"unique lemmatized words",words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


#create training data
training=[]

#create output  empty array
output_=[0]*len(classes)


#bag of words algorithm
for doc in documents:
    #intialize bag for each word
    bag=[]

    pattern_word=doc[0]

    #lemmatize and lower each word in pattern_word
    pattern_word=[lemmatizer.lemmatize(word.lower()) for word in pattern_word]

    for w in words:
      bag.append(1) if w in pattern_word else bag.append(0)

    #output
    output_row=list(output_)
    output_row[classes.index(doc[1])]=1
    training.append([bag,output_row])

#shuffle features and turn into array
random.shuffle(training)
#train_x is input and train_y is output
train_x = np.array([np.array(bag) for bag, _ in training])
train_y = np.array([np.array(output_row) for _, output_row in training])


print('training data created')

#building sequential model with 3 layers

model=Sequential()
#first layer with 128 neurons
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
#second layer with 64 neurons
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
#output layer
model.add(Dense(len(train_y[0]),activation='softmax'))
sgd=SGD(lr=0.01,momentum=0.9,nesterov=True)
#compiling model
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

#fitting model
hist=model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1,validation_split=0.2)
#saving model
model.save('chatbot_model.h5',hist)

print('model created')
#loading words and classes files

words=pickle.load(open("words.pkl","rb"))
classes=pickle.load(open("classes.pkl","rb"))
def clean_up_sentence(sentence):
  #toknize sentence
  sentence_words=nltk.word_tokenize(sentence)
  #lemmtaized and lower each word
  sentence_words=[lemmatizer.lemmatize(word.lower()) for word in sentence_words]

  return sentence_words

def bow(sentence,words,show_details=False):
  #cleaning up sentence
  sentence_words=clean_up_sentence(sentence)

  bag=[0]*len(words)

  for s in sentence_words:
    for i,w in enumerate(words):
      if w==s:
        # bag equal one if current word is in the vocabulary position
        bag[i]=1

        if show_details==True:
          print('Found in %s',w)

  return np.array(bag)
def predict_class(sentence,model):
  #filter prediction that below threshold
  p=bow(sentence,words,show_details=False)
  res=model.predict(np.array([p]))[0]

  ERROR_THRESHOLD=0.25

  results=[[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

  #sort by strong probability
  results.sort(key=lambda x:x[1],reverse=True)

  return_list=[]
  for r in results:
      return_list.append({"intent":classes[r[0]],"probability":str(r[1])})


  return return_list

def getResponse(ints,intents_json):
  tag=ints[0]['intent']
  list_intents=intents_json['intents']

  for i in list_intents:

    if (tag==i['tag']):
      result=random.choice(i['responses'])
      break

  return result


def chatbot_response(msg):
  ints=predict_class(msg,model)
  res=getResponse(ints,intents)

  return res

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(sender_email, app_password, receiver_email, subject, message_body, smtp_server, smtp_port):
    try:
        # Create the email message
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = receiver_email
        message['Subject'] = subject
        message.attach(MIMEText(message_body, 'plain'))

        # Connect to the SMTP server
        smtp_server = smtplib.SMTP(smtp_server, smtp_port)
        smtp_server.starttls()  # Enable TLS encryption
        smtp_server.login(sender_email, "sibcayrqnuptaoqv")  # Use the application-specific password

        # Send the email
        smtp_server.sendmail(sender_email, receiver_email, message.as_string())
        print("Email sent successfully!")

    except Exception as e:
        print("An error occurred:", e)

    finally:
        smtp_server.quit()



from termcolor import colored
import requests


def run():
    print(colored("WELCOME TO ESKILLS CHATBOT", 'red'))
    image_url = "https://cdn.discordapp.com/attachments/1068515702216077403/1142513000058720256/1692466707197.jpg"
    width = 160  # Set your desired width in pixels
    height = 150  # Set your desired height in pixels
    print("")

    while True:
        request = input("")
        response = chatbot_response(request)
        if response == "":
         print("")
         request80 = input("Do you want to write your project description so we can send it via Email to one of our managers?  Yes / No : ")
         if request80.lower() == "yes":
            request9 = input("Please enter your project description, it will be sent via Email to one of our managers.")
            send_email("mouelhiayoub21@gmail.com", "sibcayrqnuptaoqv", "arij.bennasr@esprit.tn", "New Project Description", request9, "smtp.gmail.com", 587)

         print("Anything else we can help you with ?")


        if response == "Not sure I understand" or response == "Please give me more info" or response == "Sorry, can't understand you":
            image_response = requests.get(image_url)

            if image_response.status_code == 200:
                image_path = "temp_image.jpg"
                with open(image_path, 'wb') as img_file:
                    img_file.write(image_response.content)

                # Open the image using PIL and resize it
             

                # Display the resized image



        print(colored(response, 'cyan'))

        if request in ["goodbye", "Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"]:
            break