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
                image1_paths.append(parts[0]) # add img1 to list (first image)
                image2_paths.append(parts[1]) # add img2 to list (second image)
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

# need to complete plotting the roc curve to print out the AUC score
# and main block so it runs