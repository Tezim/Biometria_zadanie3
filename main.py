import os
import random

from deepface import DeepFace
from skimage.feature import local_binary_pattern
import cv2
import numpy
from scipy.spatial import distance as dist
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

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


# + method #1 LBP
def calculate_distances(features_1, features_2):
    distances_all = numpy.zeros((len(features_1), len(features_2)))
    for i in range(len(features_1)):
        for j in range(len(features_2)):
            distance = cosine(features_1[i], features_2[j])
            distances_all[i][j] = distance
    return distances_all


def extract_features(frame):
    lbp = local_binary_pattern(frame, 12, 1, method='uniform')
    histogram, _ = numpy.histogram(lbp.ravel(), bins=256, range=(0, 256))
    return histogram


def lbp_extractor(faces):
    pair_results = []
    for pair in faces:
        features_1 = [extract_features(frame) for frame in pair[0]]
        features_2 = [extract_features(frame) for frame in pair[1]]
        distances = calculate_distances(features_1, features_2)
        pair_results.append(distances)
    return pair_results


# + helpers
def gray_scale(video):
    grayScale = []
    for pair in video:
        pairs = []
        for video in pair:
            video_array = []
            images = video['imagesA']
            for frame in images:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                video_array.append(gray)
            pairs.append(video_array)
        grayScale.append(pairs)
    return grayScale


def readVideos(pairs):
    videos = []
    videos_dict = true_pairs if pairs else false_pairs
    for videoA, videoB in videos_dict.items():
        video_pathA = os.path.join(DIR_PATH, videoA)
        video_pathB = os.path.join(DIR_PATH, videoB)
        videos.append([numpy.load(video_pathA), numpy.load(video_pathB)])
    return videos


def showMean(videos):
    faces = []
    for video in videos:
        for frame in video:
            faces.append(frame)
    mean_face = numpy.mean(faces, axis=0)
    plt.imshow(mean_face, cmap='gray')
    plt.show()


# + //////////////////////////////////////////////////////////////////////////////////


def create_roc(true_feature_matrix, false_feature_matrix):
    threshold = 800
    # ? min value from matrix
    min_distances_T = []
    min_distances_F = []
    for x in range(len(true_feature_matrix)):
        min_distances_T.append(numpy.min(true_feature_matrix[x]))
        min_distances_F.append(numpy.min(false_feature_matrix[x]))
    # ? max value from matrix
    max_distances_T = []
    max_distances_F = []
    for x in range(len(true_feature_matrix)):
        max_distances_T.append(numpy.max(true_feature_matrix[x]))
        max_distances_F.append(numpy.max(false_feature_matrix[x]))
    # ? mean value from matrix
    mean_distances_T = []
    mean_distances_F = []
    for x in range(len(true_feature_matrix)):
        mean_distances_T.append(numpy.mean(true_feature_matrix[x]))
        mean_distances_F.append(numpy.mean(false_feature_matrix[x]))
    # ? random value from matrix
    rand_distances_T = []
    rand_distances_F = []
    for x in range(len(true_feature_matrix)):  # for each matrix
        i = random.randint(0, true_feature_matrix[x].shape[0] - 1)
        j = random.randint(0, true_feature_matrix[x].shape[1] - 1)
        rand_distances_T.append(true_feature_matrix[x][i][j])
        i = random.randint(0, false_feature_matrix[x].shape[0] - 1)
        j = random.randint(0, false_feature_matrix[x].shape[1] - 1)
        rand_distances_F.append(false_feature_matrix[x][i][j])

    distances_min = min_distances_T + min_distances_F
    labels_min = [1] * len(min_distances_T) + [0] * len(min_distances_F)
    predicted_min = [1 if num < threshold else 0 for num in distances_min]
    min_f, min_t, thresholds = roc_curve(labels_min, predicted_min)

    distances_max = max_distances_T + max_distances_F
    labels_max = [1] * len(max_distances_T) + [0] * len(max_distances_F)
    predicted_max = [1 if num < threshold else 0 for num in distances_max]
    max_f, max_t, thresholds = roc_curve(labels_max, predicted_max)

    distances_mean = mean_distances_T + mean_distances_F
    labels_mean = [1] * len(mean_distances_T) + [0] * len(mean_distances_F)
    predicted_mean = [1 if num < threshold else 0 for num in distances_mean]
    mean_f, mean_t, thresholds = roc_curve(labels_mean, predicted_mean)

    distances_rand = rand_distances_T + rand_distances_F
    labels_rand = [1] * len(rand_distances_T) + [0] * len(rand_distances_F)
    predicted_rand = [1 if num < threshold else 0 for num in distances_rand]
    rand_f, rand_t, thresholds = roc_curve(labels_rand, predicted_rand)

    numpy.savez("distances/pca50C", minT=min_distances_T, min_F=min_distances_F,
                max_T=max_distances_T, max_F=max_distances_F, mean_T=mean_distances_T, mean_F=mean_distances_F,
                rand_T=rand_distances_T, rand_F=rand_distances_F)

    plt.figure()
    plt.title("ROC PCA Threshold: {}".format(threshold))
    plt.plot(min_f, min_t, color='c', label="Min value",ls="--")
    plt.plot(max_t, max_f, color='r', label="Max value",ls="-.")
    plt.plot(mean_t, mean_f, color='k', label="Mean value",ls='dotted')
    plt.plot(rand_t, rand_f, color='g', label="Rand value",ls=':')
    plt.xlabel("False rate")
    plt.ylabel("True rate")
    plt.legend(loc="lower right")
    plt.show()


def create_PCA(videos_true_pair, videaos_false_pair):
    faces = []
    for pair in videos_true_pair:
        v1 = pair[0]
        for frame in v1:
            faces.append(frame)
        v2 = pair[1]
        for frame in v2:
            faces.append(frame)
    for pair in videos_false_pair:
        v1 = pair[0]
        for frame in v1:
            faces.append(frame)
        v2 = pair[1]
        for frame in v2:
            faces.append(frame)
    faces = numpy.stack(faces)
    mean_face = numpy.mean(faces, axis=0)
    centered_images = faces - mean_face
    flattened_images = centered_images.reshape(faces.shape[0], -1)
    pca = PCA(n_components=50)
    pca.fit(flattened_images)
    eigenfaces = pca.components_
    return eigenfaces, mean_face
    # total_eigenfaces = pca.components_.shape[0]
    # random_indices = numpy.random.choice(total_eigenfaces, size=4, replace=False)
    #
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    # for i, ax in enumerate(axes.flatten()):
    #     eigenface = pca.components_[random_indices[i]]
    #     eigenface = eigenface.reshape(256,256)  # Reshape if needed
    #     ax.imshow(eigenface, cmap='gray')
    #     ax.set_title(f"Eigenface {i + 1}")
    #     ax.axis('off')
    # plt.tight_layout()
    # plt.show()


def get_matrix(videos, eigenfaces, mean_face):
    distance_matrixes = []
    for pair in videos:
        frames1 = [frame for frame in pair[0]]
        frames2 = [frame for frame in pair[1]]
        distances_all = numpy.zeros((len(frames1), len(frames2)))
        for i in range(len(frames1)):
            for j in range(len(frames2)):
                weight1 = eigenfaces @ (frames1[i].flatten() - mean_face.flatten()).T
                weight2 = eigenfaces @ (frames2[j].flatten() - mean_face.flatten()).T
                distances_all[i][j] = numpy.linalg.norm(weight2 - weight1, axis=0)
        distance_matrixes.append(distances_all)
    return distance_matrixes





def rocs(T_min, F_min, T_max, F_max, T_mean, F_mean, T_rand, F_rand):
    threshold = 10500

    # distances_min = numpy.concatenate((T_min, F_min), axis=0)
    # labels_min = [1] * len(T_min) + [0] * len(F_min)
    # predicted_min = [1 if num < threshold else 0 for num in distances_min]
    # fpr_min, tpr_min, thresholds = roc_curve(labels_min, predicted_min)
    #
    # distances_max = numpy.concatenate((T_max, F_max), axis=0)
    # labels_max = [1] * len(T_max) + [0] * len(F_max)
    # predicted_max = [1 if num < threshold else 0 for num in distances_max]
    # fpr_max, tpr_max, thresholds = roc_curve(labels_max, predicted_max)
    #
    # distances_mean = numpy.concatenate((T_mean, F_mean), axis=0)
    # labels_mean = [1] * len(T_mean) + [0] * len(F_mean)
    # predicted_mean = [1 if num < threshold else 0 for num in distances_mean]
    # fpr_mean, tpr_mean, thresholds = roc_curve(labels_mean, predicted_mean)
    #
    # distances_rand = numpy.concatenate((T_rand, F_rand), axis=0)
    # labels_rand = [1] * len(T_rand) + [0] * len(F_rand)
    # predicted_rand = [1 if num < threshold else 0 for num in distances_rand]
    # fpr_rand, tpr_rand, thresholds = roc_curve(labels_rand, predicted_rand)
    #
    # confusion_min = confusion_matrix(labels_min, predicted_min)
    # confusion_max = confusion_matrix(labels_max, predicted_max)
    # confusion_mean = confusion_matrix(labels_mean, predicted_mean)
    # confusion_rand = confusion_matrix(labels_rand, predicted_rand)
    #
    # fig, ax = plt.subplots(figsize=(7.5, 7.5))
    # ax.matshow(confusion_min, cmap='Blues', alpha=0.3)
    # for i in range(confusion_min.shape[0]):
    #     for j in range(confusion_min.shape[1]):
    #         ax.text(x=j, y=i, s=confusion_min[i, j], va='center', ha='center', size='xx-large')
    #
    # plt.xlabel('Predictions', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix MIN', fontsize=18)
    # plt.show()
    #
    # fig, ax = plt.subplots(figsize=(7.5, 7.5))
    # ax.matshow(confusion_max, cmap='Blues', alpha=0.3)
    # for i in range(confusion_max.shape[0]):
    #     for j in range(confusion_max.shape[1]):
    #         ax.text(x=j, y=i, s=confusion_max[i, j], va='center', ha='center', size='xx-large')
    #
    # plt.xlabel('Predictions', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix MAX', fontsize=18)
    # plt.show()
    #
    # fig, ax = plt.subplots(figsize=(7.5, 7.5))
    # ax.matshow(confusion_mean, cmap='Blues', alpha=0.3)
    # for i in range(confusion_mean.shape[0]):
    #     for j in range(confusion_mean.shape[1]):
    #         ax.text(x=j, y=i, s=confusion_mean[i, j], va='center', ha='center', size='xx-large')
    #
    # plt.xlabel('Predictions', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix MEAN', fontsize=18)
    # plt.show()
    #
    # fig, ax = plt.subplots(figsize=(7.5, 7.5))
    # ax.matshow(confusion_rand, cmap='Blues', alpha=0.3)
    # for i in range(confusion_rand.shape[0]):
    #     for j in range(confusion_rand.shape[1]):
    #         ax.text(x=j, y=i, s=confusion_min[i, j], va='center', ha='center', size='xx-large')
    #
    # plt.xlabel('Predictions', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix RAND', fontsize=18)
    # plt.show()



    distances_min = numpy.concatenate((T_min, F_min), axis=0)
    tpr_min = []
    fpr_min = []
    for threshold in numpy.arange(numpy.min(distances_min), numpy.max(distances_min), 100):

        tpr_min.append(sum(distance < threshold for distance in T_min) / len(T_min))
        fpr_min.append(sum(distance < threshold for distance in F_min) / len(F_min))

    distances_max = numpy.concatenate((T_max, F_max), axis=0)
    tpr_max = []
    fpr_max = []
    for threshold in numpy.arange(numpy.min(distances_max), numpy.max(distances_max), 100):
        tpr_max.append(sum(distance < threshold for distance in T_max) / len(T_max))
        fpr_max.append(sum(distance < threshold for distance in F_max) / len(F_max))

    distances_mean = numpy.concatenate((T_mean, F_mean), axis=0)
    tpr_mean = []
    fpr_mean = []
    for threshold in numpy.arange(numpy.min(distances_mean), numpy.max(distances_mean), 100):
        tpr_mean.append(sum(distance < threshold for distance in T_mean) / len(T_mean))
        fpr_mean.append(sum(distance < threshold for distance in F_mean) / len(F_mean))

    distances_rand = numpy.concatenate((T_rand, F_rand), axis=0)
    tpr_rand = []
    fpr_rand = []
    for threshold in numpy.arange(numpy.min(distances_rand), numpy.max(distances_rand), 100):
        tpr_rand.append(sum(distance < threshold for distance in T_rand) / len(T_rand))
        fpr_rand.append(sum(distance < threshold for distance in F_rand) / len(F_rand))

    plt.figure()
    plt.title("ROC PCA iterative threshold")
    plt.plot(fpr_min, tpr_min, color='c', label="Min value",ls="--")
    plt.plot(fpr_max, tpr_max, color='r', label="Max value",ls="-.")
    plt.plot(fpr_mean, tpr_mean, color='k', label="Mean value",ls='dotted')
    plt.plot(fpr_rand, tpr_rand, color='g', label="Rand value",ls=":")
    plt.legend(loc="lower right")
    plt.xlabel("False rate")
    plt.ylabel("True rate")
    plt.show()


def dF_features(videos_pair):

    for pair in videos_pair:
        video1 = pair[0]
        frame = video1[0]
        cv2.imshow("face",frame)
        cv2.waitKey()
        embeddings = DeepFace.represent(frame,model_name="Facenet")
        print(embeddings)
    pass


if __name__ == '__main__':
    videos_true_pair = readVideos(True)
    videos_false_pair = readVideos(False)

    videos_true_pair = gray_scale(videos_true_pair)
    videos_false_pair = gray_scale(videos_false_pair)

    # * LBP

    # true_feature_matrix = lbp_extractor(videos_true_pair)
    # false_feature_matrix = lbp_extractor(videos_false_pair)

    # create_roc(true_feature_matrix, false_feature_matrix)

    # * DeepFace

    deep_matrix_T = dF_features(videos_true_pair)
    # deep_matrix_F = dF_features(videos_false_pair)

    # * PCA

    #eigenfaces, mean_face = create_PCA(videos_true_pair, videos_false_pair)
    #true_dist_matrix = get_matrix(videos_true_pair, eigenfaces, mean_face)
    #false_dist_matrix = get_matrix(videos_false_pair, eigenfaces, mean_face)

    #create_roc(true_dist_matrix, false_dist_matrix)

    #data = numpy.load("distances/pca50C.npz")

    #rocs(data['minT'], data['min_F'],data['max_T'],data['max_F'],
    #     data['mean_T'],data['mean_F'],data['rand_T'],data['rand_F'])

