from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop #, adam
from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
from PIL import Image
# import theano

import time
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import  RMSprop


# path and folder names.
master_path = '/Users/Kaito.01/Desktop/Academics/DS 4400/Project/archive/images/Images'
category_ = ['n02097658-silky_terrier', 'n02092002-Scottish_deerhound', 'n02099849-Chesapeake_Bay_retriever', 'n02091244-Ibizan_hound', 'n02095314-wire-haired_fox_terrier', 'n02091831-Saluki', 'n02102318-cocker_spaniel', 'n02104365-schipperke', 'n02090622-borzoi', 'n02113023-Pembroke', 'n02105505-komondor', 'n02093256-Staffordshire_bullterrier', 'n02113799-standard_poodle', 'n02109961-Eskimo_dog', 'n02089973-English_foxhound', 'n02099601-golden_retriever', 'n02095889-Sealyham_terrier', 'n02085782-Japanese_spaniel', 'n02097047-miniature_schnauzer', 'n02110063-malamute', 'n02105162-malinois', 'n02086079-Pekinese', 'n02097130-giant_schnauzer', 'n02113978-Mexican_hairless', 'n02107142-Doberman', 'n02097209-standard_schnauzer', 'n02115913-dhole', 'n02106662-German_shepherd', 'n02106382-Bouvier_des_Flandres', 'n02110185-Siberian_husky', 'n02094258-Norwich_terrier', 'n02093991-Irish_terrier', 'n02094114-Norfolk_terrier', 'n02109525-Saint_Bernard', 'n02093754-Border_terrier', 'n02105251-briard', 'n02108551-Tibetan_mastiff', 'n02108422-bull_mastiff', 'n02085936-Maltese_dog', 'n02093859-Kerry_blue_terrier', 'n02104029-kuvasz', 'n02107574-Greater_Swiss_Mountain_dog', 'n02095570-Lakeland_terrier', 'n02086646-Blenheim_spaniel', 'n02088238-basset', 'n02098286-West_Highland_white_terrier', 'n02085620-Chihuahua', 'n02106166-Border_collie', 'n02090379-redbone', 'n02090721-Irish_wolfhound', 'n02088632-bluetick', 'n02113712-miniature_poodle', 'n02113186-Cardigan', 'n02108000-EntleBucher', 'n02091467-Norwegian_elkhound', 'n02100236-German_short-haired_pointer', 'n02107683-Bernese_mountain_dog', 'n02086910-papillon', 'n02097474-Tibetan_terrier', 'n02101006-Gordon_setter', 'n02093428-American_Staffordshire_terrier', 'n02100583-vizsla', 'n02105412-kelpie', 'n02092339-Weimaraner', 'n02107312-miniature_pinscher', 'n02108089-boxer', 'n02112137-chow', 'n02105641-Old_English_sheepdog', 'n02110958-pug', 'n02087394-Rhodesian_ridgeback', 'n02097298-Scotch_terrier', 'n02086240-Shih-Tzu', 'n02110627-affenpinscher', 'n02091134-whippet', 'n02102480-Sussex_spaniel', 'n02091635-otterhound', 'n02099267-flat-coated_retriever', 'n02100735-English_setter', 'n02091032-Italian_greyhound', 'n02099712-Labrador_retriever', 'n02106030-collie', 'n02096177-cairn', 'n02106550-Rottweiler', 'n02096294-Australian_terrier', 'n02087046-toy_terrier', 'n02105855-Shetland_sheepdog', 'n02116738-African_hunting_dog', 'n02111277-Newfoundland', 'n02089867-Walker_hound', 'n02098413-Lhasa', 'n02088364-beagle', 'n02111889-Samoyed', 'n02109047-Great_Dane', 'n02096051-Airedale', 'n02088466-bloodhound', 'n02100877-Irish_setter', 'n02112350-keeshond', 'n02096437-Dandie_Dinmont', 'n02110806-basenji', 'n02093647-Bedlington_terrier', 'n02107908-Appenzeller', 'n02101556-clumber', 'n02113624-toy_poodle', 'n02111500-Great_Pyrenees', 'n02102040-English_springer', 'n02088094-Afghan_hound', 'n02101388-Brittany_spaniel', 'n02102177-Welsh_springer_spaniel', 'n02096585-Boston_bull', 'n02115641-dingo', 'n02098105-soft-coated_wheaten_terrier', 'n02099429-curly-coated_retriever', 'n02108915-French_bulldog', 'n02102973-Irish_water_spaniel', 'n02112018-Pomeranian', 'n02112706-Brabancon_griffon', 'n02094433-Yorkshire_terrier', 'n02105056-groenendael', 'n02111129-Leonberg', 'n02089078-black-and-tan_coonhound']
short_category = ['n02085936-Maltese_dog', 'n02088094-Afghan_hound', 'n02092002-Scottish_deerhound', 'n02112018-Pomeranian', 'n02107683-Bernese_mountain_dog', 'n02111889-Samoyed', 'n02090721-Irish_wolfhound', 'n02086240-Shih-Tzu', 'n02111500-Great_Pyrenees', 'n02111129-Leonberg'] # length 10 top 10


def train_test_split(path_, categories_):
    # Step 3: Creating training data
    training = []
    for category in categories_:
        path = os.path.join(path_, category)
        class_num = categories_.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            # new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training.append([img_array, class_num])

    # Step 4: Shuffle the Dataset
    random.shuffle(training)

    # Step 5: Assigning Labels and Features
    X =[]
    y =[]
    for features, label in training:
      X.append(features)
      y.append(label)
    X = np.array(X).reshape(-1, np.array(X).shape[0], np.array(X).shape[1], len(categories_)) # this has to be fixed

    # Step 6: Normalizing X and Converting Labels to Categorical Data
    X = X.astype('float32') # OpenCV initially stores value as uint8. Floating-point is preferrable for Neural Networks
    X /= 255 # X will now be floating point between 0 and 1, easier to categorize
    Y = np_utils.to_categorical(y, categories_) # One Hot Encoding
    print(Y[100])
    print(Y.shape)

    # Step 7: Split X and Y for Use in CNN
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.25, # 75:25 training:testing ratio
        random_state = len(categories_)
    )

    return [X_train, X_test, y_train, y_test]

# lecture-provided model
# we need to tune the Dense layers
def init_model(categories_):
    start_time = time.time()

    print("Compiling Model")
    model = Sequential()
    model.add(Dense(10, input_dim=len(categories_)))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

    print("Model finished:"+format(time.time() - start_time))
    return model

def run_network(data, model=None, epochs=20, batch=256):
    try:
        start_time = time.time()
        X_train, X_test, y_train, y_test = data

        print("Training model")
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_test, y_test), verbose=2)

        print("Training duration:"+format(time.time()-start_time))
        score = model.evaluate(X_test, y_test, batch_size=16)

        print("\nNetwork's test loss and accuracy:"+format(score))
        return history
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        return history

def plot_losses(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()
