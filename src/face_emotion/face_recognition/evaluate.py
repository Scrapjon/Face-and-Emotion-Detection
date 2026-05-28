import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc # roc curv and aux will take the labels and scores to calculate false/true postive rates
from deepface import DeepFace 

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

def get_similarity(img1, img2):
    result = DeepFace.verify(
        img1_path=img1,
        img2_path=img2,
        model_name='Facenet', # use facenet model for face recognition same as in model.py
        detector_backend='opencv', # match the detector too
        distance_metric='cosine', # match model.py
        enforce_detection=False, # if no face is detected, just return a similarity score of 0, so it wont crash if there's no face
        silent=True # no need to print stuff to console when calculating similarity
    )
    return 1 - result['distance'] # 0 would mean identical, 1 means different in distance. ROC curve will need simililarity (higher more similar) so we do 1 - distance to convert it to similarity score
# distance turns 0 into 1, and 1 into 0 to match the scales

def compute_all_scores(img1_list, img2_list):
    scores = []
    for i, (img1, img2) in enumerate(zip(img1_list, img2_list)): # use zip to join the lists together into pairs, enumerate to give counters to each pair
        try:
            score = get_similarity(img1, img2)
        except Exception:
            score = 0 # if the pair fails then give it as 0
        scores.append(score)
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

    print("Loading pairs..")
    img1_list, img2_list, labels = load_pairs(pairs_file)

    print(f"Loaded {len(labels)} pairs. Computing similarity scores...")
    scores = compute_all_scores(img1_list, img2_list) # go through each pair through deepface and get similiarity scores

    print("Plotting ROC Cure..")
    plot_roc_curve(labels, scores)
