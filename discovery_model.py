import numpy as np
import cv2
import os
import random
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt
import time
from keras.models import Sequential
import tensorflow as tf
import json

IMG_SIZE = 200  # Define the image size

def crop_image(image, crop_size):
    # Crop image to specified size
    h, w = image.shape[:2]
    if h > crop_size[0] and w > crop_size[1]:
        return image[(h - crop_size[0]) // 2:(h + crop_size[0]) // 2, (w - crop_size[1]) // 2:(w + crop_size[1]) // 2]
    else:
        return cv2.resize(image, tuple(crop_size))

def split(path_, categories_):
    training = []
    for category in categories_:
        path = os.path.join(path_, category)
        class_num = categories_.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_array = crop_image(img_array, [IMG_SIZE, IMG_SIZE])
            training.append([new_array, class_num])

    random.shuffle(training)

    X = []
    y = []
    for features, label in training:
        X.append(features)
        y.append(label)
    
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    X = X.astype('float32') / 255.0
    Y = to_categorical(y, len(categories_))
    y = np.array(y).reshape(-1,)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=len(categories_)
    )

    return [X_train, X_test, y_train, y_test]

def init_model(categories_):
    model = Sequential([
        Convolution2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=2),
        Convolution2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=2),
        Dropout(0.9),
        Flatten(),
        Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.8),
        Dense(len(categories_), activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 10
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

def run_network(data, model=None, epochs=25, batch=4):
    try:
        start_time = time.time()
        X_train, X_test, y_train, y_test = data
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        lr_callback = LearningRateScheduler(lr_scheduler)

        print("Training model")
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_test, y_test), verbose=2, callbacks=[early_stopping, lr_callback])

        print("Training duration:", time.time()-start_time)
        score = model.evaluate(X_test, y_test, batch_size=16)

        print("\nNetwork's test loss and accuracy:", score)
        return history, score
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        return None, None

def save_results(history, score):
    results = {
        'history': history.history,
        'test_loss': score[0],
        'test_accuracy': score[1]
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f)

def plot_losses(hist):
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(hist.history['loss'])
    axs[0].plot(hist.history['val_loss'])
    axs[0].set_title('Model Loss')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Test'], loc='upper right')

    axs[1].plot(hist.history['accuracy'])
    axs[1].plot(hist.history['val_accuracy'])
    axs[1].set_title('Model Accuracy')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Test'], loc='upper right')

    plt.show()


master_path = '/images/Images'
category_ = ['n02097658-silky_terrier', 'n02092002-Scottish_deerhound', 'n02099849-Chesapeake_Bay_retriever', 'n02091244-Ibizan_hound', 'n02095314-wire-haired_fox_terrier', 'n02091831-Saluki', 'n02102318-cocker_spaniel', 'n02104365-schipperke', 'n02090622-borzoi', 'n02113023-Pembroke', 'n02105505-komondor', 'n02093256-Staffordshire_bullterrier', 'n02113799-standard_poodle', 'n02109961-Eskimo_dog', 'n02089973-English_foxhound', 'n02099601-golden_retriever', 'n02095889-Sealyham_terrier', 'n02085782-Japanese_spaniel', 'n02097047-miniature_schnauzer', 'n02110063-malamute', 'n02105162-malinois', 'n02086079-Pekinese', 'n02097130-giant_schnauzer', 'n02113978-Mexican_hairless', 'n02107142-Doberman', 'n02097209-standard_schnauzer', 'n02115913-dhole', 'n02106662-German_shepherd', 'n02106382-Bouvier_des_Flandres', 'n02110185-Siberian_husky', 'n02094258-Norwich_terrier', 'n02093991-Irish_terrier', 'n02094114-Norfolk_terrier', 'n02109525-Saint_Bernard', 'n02093754-Border_terrier', 'n02105251-briard', 'n02108551-Tibetan_mastiff', 'n02108422-bull_mastiff', 'n02085936-Maltese_dog', 'n02093859-Kerry_blue_terrier', 'n02104029-kuvasz', 'n02107574-Greater_Swiss_Mountain_dog', 'n02095570-Lakeland_terrier', 'n02086646-Blenheim_spaniel', 'n02088238-basset', 'n02098286-West_Highland_white_terrier', 'n02085620-Chihuahua', 'n02106166-Border_collie', 'n02090379-redbone', 'n02090721-Irish_wolfhound', 'n02088632-bluetick', 'n02113712-miniature_poodle', 'n02113186-Cardigan', 'n02108000-EntleBucher', 'n02091467-Norwegian_elkhound', 'n02100236-German_short-haired_pointer', 'n02107683-Bernese_mountain_dog', 'n02086910-papillon', 'n02097474-Tibetan_terrier', 'n02101006-Gordon_setter', 'n02093428-American_Staffordshire_terrier', 'n02100583-vizsla', 'n02105412-kelpie', 'n02092339-Weimaraner', 'n02107312-miniature_pinscher', 'n02108089-boxer', 'n02112137-chow', 'n02105641-Old_English_sheepdog', 'n02110958-pug', 'n02087394-Rhodesian_ridgeback', 'n02097298-Scotch_terrier', 'n02086240-Shih-Tzu', 'n02110627-affenpinscher', 'n02091134-whippet', 'n02102480-Sussex_spaniel', 'n02091635-otterhound', 'n02099267-flat-coated_retriever', 'n02100735-English_setter', 'n02091032-Italian_greyhound', 'n02099712-Labrador_retriever', 'n02106030-collie', 'n02096177-cairn', 'n02106550-Rottweiler', 'n02096294-Australian_terrier', 'n02087046-toy_terrier', 'n02105855-Shetland_sheepdog', 'n02116738-African_hunting_dog', 'n02111277-Newfoundland', 'n02089867-Walker_hound', 'n02098413-Lhasa', 'n02088364-beagle', 'n02111889-Samoyed', 'n02109047-Great_Dane', 'n02096051-Airedale', 'n02088466-bloodhound', 'n02100877-Irish_setter', 'n02112350-keeshond', 'n02096437-Dandie_Dinmont', 'n02110806-basenji', 'n02093647-Bedlington_terrier', 'n02107908-Appenzeller', 'n02101556-clumber', 'n02113624-toy_poodle', 'n02111500-Great_Pyrenees', 'n02102040-English_springer', 'n02088094-Afghan_hound', 'n02101388-Brittany_spaniel', 'n02102177-Welsh_springer_spaniel', 'n02096585-Boston_bull', 'n02115641-dingo', 'n02098105-soft-coated_wheaten_terrier', 'n02099429-curly-coated_retriever', 'n02108915-French_bulldog', 'n02102973-Irish_water_spaniel', 'n02112018-Pomeranian', 'n02112706-Brabancon_griffon', 'n02094433-Yorkshire_terrier', 'n02105056-groenendael', 'n02111129-Leonberg', 'n02089078-black-and-tan_coonhound']
short_category = ['n02085936-Maltese_dog', 'n02088094-Afghan_hound', 'n02092002-Scottish_deerhound', 'n02112018-Pomeranian', 'n02107683-Bernese_mountain_dog', 'n02111889-Samoyed', 'n02090721-Irish_wolfhound', 'n02086240-Shih-Tzu', 'n02111500-Great_Pyrenees', 'n02111129-Leonberg'] # length 10 top 10
IMG_SIZE = 300

data = split(master_path, category_)
model = init_model(category_)
history, score = run_network(data, model)

plot_losses(history)
save_results(history, score)
