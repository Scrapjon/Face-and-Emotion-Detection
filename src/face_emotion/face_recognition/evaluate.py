import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc # roc curv and aux will take the labels and scores to calculate false/true postive rates
# removed deepface and replaced with custom model   
from tensorflow.keras.models import Model  # lets us build/load keras models
import cv2  # lets us read and resize images

# ADDING NEW CUSTOM MODEL  
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def load_vggface_model(weights_path):
    from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, Activation

    model = Sequential() # faced an issue calling deepface.verify, should automatically create the weights and architecture however when using our own fine-turned weights, need an empty shell of the architecture deepface wont permit to swap out easily afer doing research

    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation("softmax"))

    model.load_weights(weights_path)
    model.trainable = False

    # Output from the 4096-dim embedding layer, not the final softmax
    embedding_model = Model(inputs=model.input, outputs=model.layers[-4].output)
    return embedding_model

def get_embedding(model, img_path):
    img = cv2.imread(img_path)  # read image from disk
    img = cv2.resize(img, (224, 224))  # resize to what VGGFace expects
    img = img.astype("float32") / 255.0  # convert pixels from 0-255 to 0-1
    img = np.expand_dims(img, axis=0)  # add batch dimension so model accepts it (224,224,3) to (1,224,224,3)
    return model.predict(img, verbose=0)[0]  # run through model, return the 4096-length embedding vector


def load_pairs(pairs_file): # takes txt file as input
    image1_paths, image2_paths, labels = [], [], [] # 3 empty lists to store data
    with open(pairs_file, 'r') as f: # open in read mode
        for line in f: # loop every line one at a time
            parts = line.strip().split() # splits 'img1 img2 1into a list
            if len(parts) == 3:
                image1_paths.append("data/" + parts[0]) # add img1 to list (first image)
                image2_paths.append("data/" + parts[1]) # add img2 to list (second image)
                labels.append(int(parts[2])) # add label (1 or 0) to list, 1 means same person, 0 means different, need int to convert string to number
    return image1_paths, image2_paths, labels

def get_similarity(model, img1, img2):
    emb1 = get_embedding(model, img1)  # turn image 1 into an embedding vector
    emb2 = get_embedding(model, img2)  # turn image 2 into an embedding vector
    dot = np.dot(emb1, emb2)  # multiply the two vectors together (measures how aligned they are)
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)  # get the size/length of each vector
    return dot / (norm + 1e-8)  # divide to get similarity score between 0 and 1, +1e-8 stops dividing by zero

def compute_all_scores(model, img1_list, img2_list):
    scores = []
    for i, (img1, img2) in enumerate(zip(img1_list, img2_list)): # use zip to join the lists together into pairs, enumerate to give counters to each pair
        try:
            score = get_similarity(model, img1, img2)
        except Exception:
            score = 0 # if the pair fails then give it as 0
        scores.append(score)
        if (i + 1) % 100 == 0:
            print(f"Pair {i+1}/{len(img1_list)} done") #i starts at 0 so add 1 to make it start at 1, and print progress to console
    return scores

def plot_roc_curve(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores) # use sklearn to calculate the false positive and true positive rates from labels and scores
    roc_auc = auc(fpr, tpr) # calculates the AUC score from the false/true positive rates (higher is better)

    plt.figure() # blank page to draw on
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})') # draw the ROC curve in blue and have the AUC score in the legend
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess') # create a diagonal line to show how random guessing looks like
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Face Verification')
    plt.legend()
    plt.tight_layout() # adjust spacing so nothing gets cut off
    plt.savefig('roc_curve.png') # save file
    plt.show()
    print(f"AUC Score: {roc_auc:.4f}") # print AUC score to 4 decimal places
    return roc_auc # return the AUC score 

    # added main
if  __name__ == "__main__": # makes sure its ran only if the file is run directly 
    pairs_file = "data/verification_pairs_val.txt"    # path to the pairs txt file

    print("Loading model...")
    model = load_vggface_model("src/face_emotion/face_recognition/vggFineTuning/new_weights.weights.h5")

    print("Loading pairs..")
    img1_list, img2_list, labels = load_pairs(pairs_file)

    print(f"Loaded {len(labels)} pairs. Computing similarity scores...")
    scores = compute_all_scores(model, img1_list, img2_list) # go through each pair through deepface and get similiarity scores

    np.save("scores.npy", np.array(scores))
    np.save("labels.npy", np.array(labels))

    print("Plotting ROC Cure..")
    plot_roc_curve(labels, scores)
