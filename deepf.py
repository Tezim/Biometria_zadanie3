import os
import random
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from deepface import DeepFace
from deepface.commons import distance as dst, functions
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import cv2

DIR_PATH = "data"
true_pairs = {
    'Karin_Viard_1.npz': 'Karin_Viard_3.npz',
    'Kate_Winslet_0.npz': 'Kate_Winslet_1.npz',
    'Laura_Bush_2.npz': 'Laura_Bush_4.npz',
    'Keith_Olbermann_0.npz': 'Keith_Olbermann_1.npz',
    'Kemal_Dervis_1.npz': 'Kemal_Dervis_0.npz',
    'Kenneth_Reichert_3.npz': 'Kenneth_Reichert_2.npz',
    'Kevin_Spacey_1.npz': 'Kevin_Spacey_0.npz',
    'Kirsten_Dunst_3.npz': 'Kirsten_Dunst_0.npz',
    'Nora_Ephron_4.npz': 'Nora_Ephron_1.npz',
    'Orlando_Bloom_5.npz': 'Orlando_Bloom_0.npz'
}
false_pairs = {
    'Karin_Viard_1.npz': 'Kim_Cattrall_0.npz',
    'Kate_Winslet_0.npz': 'Natasha_Henstridge_1.npz',
    'Oscar_Elias_Biscet_5.npz': 'Karin_Viard_1.npz',
    'Matt_LeBlanc_3.npz': 'Oscar_Elias_Biscet_5.npz',
    'Kristy_Curry_0.npz': 'Olympia_Dukakis_4.npz',
    'Lee_Baca_0.npz': 'Martin_Landau_0.npz',
    'Noel_Niell_4.npz': 'Kathryn_Morris_2.npz',
    'Maria_Callas_2.npz': 'Noel_Niell_4.npz',
    'Kevin_Spacey_2.npz': 'Mark_Kelly_0.npz',
    'Kathryn_Morris_2.npz': 'Nina_Jacobson_3.npz'
}

worst = {
    'Noel_Niell_4.npz': 'Kathryn_Morris_2.npz',
    'Karin_Viard_1.npz': 'Kim_Cattrall_0.npz'
}

def get_features(frame):
    embedding = None
    target_size = functions.find_target_size(model_name="SFace")
    image_object = functions.extract_faces(
        img=frame,
        target_size=target_size,
        detector_backend="opencv",
        grayscale=False,
        enforce_detection=False,
        align=False
    )
    feature = DeepFace.represent(img_path=image_object[0][0], model_name="SFace", detector_backend="skip")
    try:
        embedding = feature[0].get('embedding')
    except (IndexError, AttributeError, TypeError):
        pass
    return embedding


def get_distances(f1, f2, which):
    return dst.findEuclideanDistance(f1, f2) if which else dst.findCosineDistance(f1, f2)


def compare(video1, video2, method):
    features1 = []
    features2 = []

    for frame in video1:
        features1.append(get_features(frame))
    for frame in video2:
        features2.append(get_features(frame))

    matrix = np.zeros((len(features1), len(features2)))
    for i in range(len(features1)):
        for j in range(len(features2)):
            distance = get_distances(features1[i], features2[j], method)
            matrix[i][j] = distance
    return matrix, features1, features2


def extract_features(true_videos, false_videos, metrics):
    true_matrix = []
    true_features = []
    false_features = []
    false_matrix = []
    for pair in true_videos:
        video1 = pair[0]
        video2 = pair[1]
        matrix, f1, f2 = compare(video1, video2, metrics)
        true_matrix.append(matrix)
        true_features.append(f1)
        true_features.append(f2)
    for pair in false_videos:
        video1 = pair[0]
        video2 = pair[1]
        matrix, f1, f2 = compare(video1, video2, metrics)
        false_matrix.append(matrix)
        false_features.append(f1)
        false_features.append(f2)
    return true_matrix, false_matrix, true_features, false_features


def plot_roc(model):
    data = pd.read_csv(model + '-result.csv')
    true_labels = [1 if item == 0 else 0 for item in data['expected_result']]
    for i in ['min_value', 'max_value', 'mean_value', 'random_value']:
        predicted = data[i]
        fpr, tpr, thresholds = roc_curve(true_labels, predicted)
        auc = roc_auc_score(true_labels, predicted)
        plt.plot(fpr, tpr, label=i + ' (AUC = {:.2f})'.format(auc))
        best_threshold_index = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_threshold_index]
        print(f'Best Threshold for {i}: ', best_threshold)

    plt.plot([0, 1], [0, 1], linestyle='--', label='Random (ROC=0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for ' + model)
    plt.legend()
    plt.show()


def make_rocs(data, name):
    true_labels = [1] * 10 + [0] * 10

    min_vals = data[0]
    max_vals = data[1]
    mean_vals = data[2]
    rand_vals = data[3]

    fpr_min, tpr_min, thresholds = roc_curve(true_labels, min_vals)
    auc_min = roc_auc_score(true_labels, min_vals)
    fpr_max, tpr_max, thresholds = roc_curve(true_labels, max_vals)
    auc_max = roc_auc_score(true_labels, max_vals)
    fpr_mean, tpr_mean, thresholds = roc_curve(true_labels, mean_vals)
    auc_mean = roc_auc_score(true_labels, mean_vals)
    fpr_rand, tpr_rand, thresholds = roc_curve(true_labels, rand_vals)
    auc_rand = roc_auc_score(true_labels, rand_vals)

    plt.figure(1)
    plt.title("ROC SFace {}".format(name))
    plt.xlabel("False rate")
    plt.ylabel("True rate")
    plt.plot(tpr_min, fpr_min, color='c', label="Min Values: {}".format(auc_min), ls="--")
    plt.plot(tpr_max, fpr_max, color='r', label="Max Values: {}".format(auc_max), ls=":")
    plt.plot(tpr_mean, fpr_mean, color='b', label="Mean Values: {}".format(auc_mean), ls="-.")
    plt.plot(tpr_rand, fpr_rand, color='k', label="Rand Values: {}".format(auc_rand), ls=":")
    plt.plot([0,1],[0,1], color='g',label="X=Y")
    plt.legend(loc="lower right")
    plt.show()

    labels_m = []
    for threshold in range(34, 45,):
        min_converted = [1 if x < (threshold/100) else 0 for x in min_vals]
        labels_m.append(min_converted)


    cm = confusion_matrix(true_labels, min_converted)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cm, cmap='Blues', alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix Min', fontsize=18)
    plt.show()


    print()


def evaluate(T_matrix, F_matrix):
    _min = []
    _max = []
    _mean = []
    _rand = []
    for matrix in T_matrix:
        _min.append(numpy.min(matrix))
        _max.append(numpy.max(matrix))
        _mean.append(numpy.mean(matrix))
        _rand.append(matrix[random.randint(0, len(matrix) - 1)][random.randint(0, len(matrix[0]) - 1)])
    for matrix in F_matrix:
        _min.append(numpy.min(matrix))
        _max.append(numpy.max(matrix))
        _mean.append(numpy.mean(matrix))
        _rand.append(matrix[random.randint(0, len(matrix) - 1)][random.randint(0, len(matrix[0]) - 1)])
    return [_min, _max, _mean, _rand]


def readVideos(pairs):
    videos = []
    if pairs == 0:
        videos_dict = true_pairs
    elif pairs == 1:
        videos_dict = false_pairs
    else:
        videos_dict = worst
    for videoA, videoB in videos_dict.items():
        video_pathA = os.path.join(DIR_PATH, videoA)
        video_pathB = os.path.join(DIR_PATH, videoB)
        videos.append([numpy.load(video_pathA), numpy.load(video_pathB)])
    return videos


def gray_scale(video):
    grayScale = []
    for pair in video:
        pairs = []
        for video in pair:
            video_array = []
            images = video['imagesA']
            for frame in images:
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                video_array.append(frame)
            pairs.append(video_array)
        grayScale.append(pairs)
    return grayScale


def show_worst():
    worst = readVideos(2)
    worst = gray_scale(worst)
    for pair in worst:
        video1 = pair[0]
        video2 = pair[1]
        cv2.imshow("Frame 1 v1", video1[0])
        cv2.waitKey(0)
        cv2.imshow("Frame 1 v2", video2[0])
        cv2.waitKey(0)

    print()


if __name__ == '__main__':
    videos_true_pair = readVideos(1)
    videos_false_pair = readVideos(0)
    #
    videos_true_pair = gray_scale(videos_true_pair)
    videos_false_pair = gray_scale(videos_false_pair)
    #
    true_matrix_cos, false_matrix_cos, true_features_cos, false_features_cos = extract_features(videos_true_pair, videos_false_pair, 0)
    #true_matrix_euclid, false_matrix_euclid = extract_features(videos_true_pair, videos_false_pair, 1)
    numpy.savez("distances/true_features", fatures=true_features_cos)
    numpy.savez("distances/false_features", fatures=false_features_cos)



    #
    # results_cos = evaluate(true_matrix_cos, false_matrix_cos)
    # results_euclid = evaluate(true_matrix_euclid, false_matrix_euclid)
    #
    # numpy.savez("distances/SFACE_cosinus", arrays=results_cos)
    # numpy.savez("distances/SFACE_euclidean", arrays=results_euclid)

    # distances_cos = numpy.load("distances/SFACE_cosinus.npz")['arrays']
    # distances_euclid = numpy.load("distances/SFACE_euclidean.npz")['arrays']

    # make_rocs(distances_cos, "Cos...")
    # make_rocs(distances_euclid, "Eclid...")

    show_worst()
