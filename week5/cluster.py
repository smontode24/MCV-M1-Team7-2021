
import glob
import cv2
import sys
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage import feature 
import os
import shutil
from tqdm import tqdm
import time
from descriptors import DCT, twod_hist, celled_2dhist, mrhm
from metrics import l2_dist

def save_rep_imgs(top_k_imgs, kmeans, name):
    for c_class, idx_rep in enumerate(top_k_imgs):
        saved_imgs = 0
        for num_rep, idx in enumerate(idx_rep):
            if kmeans.labels_[idx] == c_class:
                dst_dir = '/home/sergio/MCV/M1/Practicas/DB/Clustering/'+ name + str(c_class)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                shutil.copy(filenames[idx], os.path.join(dst_dir, "rep_"+str(saved_imgs)+".jpg"))
                saved_imgs += 1
                if saved_imgs == 5:
                    break

def dct_clustering(images, n_clusters=10, top_representative=5):

    museum_histograms = []
    museum_textures = []

    for museum_image in tqdm(images):
        museum_image = cv2.imread(museum_image)
        museum_textures.append(DCT(museum_image, num_blocks=12))

    feat_np = np.asarray(museum_textures, dtype=np.float64)
    print("\n Performing KMeans")
    kmeans = KMeans(n_clusters=n_clusters, random_state=10).fit(feat_np)

    for i, k_l in enumerate(kmeans.labels_):
        dst_dir = '/home/sergio/MCV/M1/Practicas/DB/Clustering/DCT' + str(k_l)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.copy(filenames[i], dst_dir)

    dist_matrix = l2_dist(kmeans.cluster_centers_, np.array(museum_textures))
    top_k_imgs = np.argsort(dist_matrix)
    save_rep_imgs(top_k_imgs, kmeans, "DCT")

def dct_clustering_color(images, n_clusters=10, top_representative=5):

    museum_histograms = []
    museum_textures = []

    for museum_image in tqdm(images):
        museum_image = cv2.imread(museum_image)
        museum_textures.append(np.concatenate([DCT(museum_image, num_blocks=12), celled_2dhist(museum_image, [4,4])]))

    feat_np = np.asarray(museum_textures, dtype=np.float64)
    print("\n Performing KMeans")
    kmeans = KMeans(n_clusters=n_clusters, random_state=10).fit(feat_np)

    for i, k_l in enumerate(kmeans.labels_):
        dst_dir = '/home/sergio/MCV/M1/Practicas/DB/Clustering/DCT_color' + str(k_l)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.copy(filenames[i], dst_dir)

    dist_matrix = l2_dist(kmeans.cluster_centers_, np.array(museum_textures))
    top_k_imgs = np.argsort(dist_matrix)

    save_rep_imgs(top_k_imgs, kmeans, "DCT_color")

def bow_clustering(images, n_clusters=10, top_representative=5):

    museum_histograms = []
    museum_textures = []

    dico = []
    for museum_image in tqdm(images):
        museum_image = cv2.imread(museum_image)

        grayscale_image = cv2.cvtColor(museum_image, cv2.COLOR_BGR2GRAY)
        grayscale_image = cv2.resize(grayscale_image, (256, 256), interpolation=cv2.INTER_AREA)
        orb = cv2.ORB_create(nfeatures=500, fastThreshold=15, scaleFactor=1.2) 
        kp, des = orb.detectAndCompute(grayscale_image, None)
        if des is not None:
            for d in des:
                dico.append(d)
    
    k = 256
    batch_size = len(images) * 3
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(dico)

    print("obtaining histograms")
    hists = obtain_hist_bow(images, k, kmeans)

    print("Clustering similar images by BOW")
    feat_np = np.asarray(hists, dtype=np.float64)
    kmeans = KMeans(n_clusters=n_clusters, random_state=10).fit(feat_np)

    for i, k_l in enumerate(kmeans.labels_):
        dst_dir = '/home/sergio/MCV/M1/Practicas/DB/Clustering/BOW' + str(k_l)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.copy(filenames[i], dst_dir)

    dist_matrix = l2_dist(kmeans.cluster_centers_, np.array(hists))
    top_k_imgs = np.argsort(dist_matrix)

    save_rep_imgs(top_k_imgs, kmeans, "BOW")
    
def obtain_hist_bow(images, n_clusters, kmeans):
    histo_list = []

    for leaf in tqdm(images):
        img = cv2.imread(leaf) 
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayscale_image = cv2.resize(grayscale_image, (256, 256), interpolation=cv2.INTER_AREA)
        orb = cv2.ORB_create(nfeatures=100, fastThreshold=15, scaleFactor=1.2) 
        kp, des = orb.detectAndCompute(grayscale_image, None)

        histo = np.zeros(n_clusters)
        nkp = np.size(kp)

        if des is not None:
            for d in des:
                idx = kmeans.predict([d])
                histo[idx] += 1 # Because we need normalized histograms, I prefere to add 1/nkp directly

        histo_list.append(histo)

    return np.array(histo_list)

def color_clustering(images, n_clusters=10, top_representative=5):

    museum_histograms = []
    museum_textures = []

    for museum_image in tqdm(images):
        museum_image = cv2.imread(museum_image)
        museum_textures.append(mrhm(museum_image, num_blocks=12))

    feat_np = np.asarray(museum_textures, dtype=np.float64)
    print("\n Performing KMeans")
    kmeans = KMeans(n_clusters=n_clusters, random_state=10).fit(feat_np)

    for i, k_l in enumerate(kmeans.labels_):
        dst_dir = '/home/sergio/MCV/M1/Practicas/DB/Clustering/color' + str(k_l)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.copy(filenames[i], dst_dir)

    dist_matrix = l2_dist(kmeans.cluster_centers_, np.array(museum_textures))
    top_k_imgs = np.argsort(dist_matrix)

    save_rep_imgs(top_k_imgs, kmeans, "color")

if __name__ == "__main__":
    filenames = glob.glob("/home/sergio/MCV/M1/Practicas/DB/BBDD/*.jpg")
    filenames.sort()

    print("Starting clustering\n")
    time_start = time.time()
    bow_clustering(filenames)
    print("Time to cluster: ", time.time() - time_start)
