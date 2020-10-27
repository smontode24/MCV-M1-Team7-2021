import glob
import cv2
import numpy
import os

def path1 ():
    path = "/home/ubuntu/PycharmProjects/MCVTeam7/week2/qsd1_w2/*.txt"

    return path
def path2 ():
    path = "/home/ubuntu/PycharmProjects/MCVTeam7/week2/BBDD"

    return path

def list_files_qs(path):

    for directory, subdirectories, files in os.walk(path):
        for file in files:
            f = open(file, 'r')
            f_contents = f.readline()
            f_contents = str(f_contents)
            f.close()
            f2 = open('List.txt', 'a')
            f2.write(f_contents)
            f2.close()
    return 0

def list_files_BBDD(path):

    for directory, subdirectories, files in os.walk(path):
        for file in files:
            # Si fas un open, directe del fitxer,
            # que conté el nom del fitxer.... l'hauries d'anexar al path inicial
            # a no ser que estigues al mateix lloc
            # El programa petara (al no trobar en el lloc on s'executa)
            # el fitxer "bbdd_000001.png"
            # FIX 1:
            # afegir nom del fitxer al path original
            full_path = os.path.join(path, file)
            f = open(full_path, 'r')
            f_contents = f.readline()
            f_contents = str(f_contents)
            f.close()
            f2 = open('List_BBDD.txt', 'a')
            f2.write(f_contents)
            f2.close()
    return 0

def compare_lists ():
    file1 = open('List.txt', 'r')
    for line in file1:
        f_contents = file1.readline()
        f_contents = str(f_contents)
        file2 = open('List_BBDD.txt', 'r')
        for line in file2:
            f2_contents = file2.readline()
            f2_contents = str(f2_contents)
            if f_contents == f2_contents:
                compare_list = open('Matchs.txt', 'a')
                compare_list.write(f_contents)
                compare_list.close()

if __name__ == "__main__":

    path = path1()
    path2 = path2()
    list_BBDD = list_files_BBDD(path2)
    list = list_files_qs(path)
    compare_lists()
