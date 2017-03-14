import cv2
import numpy
import random




def main():
    MAX_LARGEUR = 400
    MAX_HAUTEUR = 400

    K = 3 #Le fameux parametre K de l'algorithme



    # Charger l'image et la reduire si trop grande (sinon, on risque de passer trop de temps sur le calcul...)
    imagecolor = cv2.imread('perr.jpg')
    if imagecolor.shape[0] > MAX_LARGEUR or imagecolor.shape[1] > MAX_HAUTEUR:
        factor1 = float(MAX_LARGEUR) / imagecolor.shape[0]
        factor2 = float(MAX_HAUTEUR) / imagecolor.shape[1]
        factor = min(factor1, factor2)
        imagecolor = cv2.resize(imagecolor, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)



    # Le nombre de pixels de l'image
    nb_pixels = imagecolor.shape[0] * imagecolor.shape[1]


    # On affiche une fenetre avec l'image
    cv2.namedWindow("image")
    #On sort quand l'utilisateur appuie sur une touche
    cv2.imshow("image", imagecolor)
    key = cv2.waitKey(0)


    #Les coordonnees BRV de tous les pixels de l'image (les elements de E)
    bleu = imagecolor[:, :, 0].reshape(nb_pixels, 1)
    vert = imagecolor[:, :, 1].reshape(nb_pixels, 1)
    rouge = imagecolor[:, :, 2].reshape(nb_pixels, 1)




    #Les coordonnees BRV de chaque point-cluster (les elements de N)
    cluster_bleu = numpy.zeros(K)
    cluster_vert = numpy.zeros(K)
    cluster_rouge = numpy.zeros(K)


    #Ce tableau permet de connaitre, pour chaque pixel de l'image, a quel cluster il appartient
    #On le remplit au hasard
    groupe = numpy.zeros((nb_pixels, 1)) #groupe est un tableau de Card(E) cases, et chaque valeur est un entier entre 0 et K-1, designant le cluster auquel chaque point sera rattache
    #On remplit au hasard le tableau groupe, c'est a dire que l'on attribue au hasard chaque point de l'espace a un des K clusters
    #Cependant, pour etre sur qu'au depart chaque cluster est rattache a au moins un point de l'espace, on attribue les K premiers points de l'espace a chaque K clusters
    for i in range(0,K):
        groupe[i,0]=i
    #La, on fait l'attribution du reste des points de l'espace a des clusters choisis au hasard
    for i in range(K,nb_pixels):
        groupe[i,0]=random.randint(0, K-1)




    #La, c'est a vous d'ecrire le code de la boucle principale
    #Votre code doit faire evoluer les tableaux groupe, cluster_bleu, cluster_rouge et cluster_vert
    #...
    # permet de boucler tant que les clusters changent
    there_are_no_modification = 0

    print "Nombre de linges et colonnes"
    print imagecolor.shape[0]
    print imagecolor.shape[1]

    # Boucle tant que les clusters changent
    while( there_are_no_modification == 0 ):
        old_cluster_bleu = numpy.copy(cluster_bleu)
        old_cluster_vert = numpy.copy(cluster_vert)
        old_cluster_rouge = numpy.copy(cluster_rouge)

        # Calcul des positions des nouveaux points cluster
        for i in range(0,K):
            idx = (groupe[:,0] == i)
            barycentre_rouge = 0
            barycentre_vert = 0
            barycentre_bleu = 0

            taille_du_groupe = numpy.count_nonzero(idx == 1)

            print "Taille du groupe = "
            print taille_du_groupe

            # Calcul du barycentre
            for j in range( 0,taille_du_groupe ):
                y = j%imagecolor.shape[1]
                x = int(int(j)/int(imagecolor.shape[1]))
                barycentre_rouge += imagecolor[ x, y, 2 ]
                barycentre_vert += imagecolor[ x, y, 1 ]
                barycentre_bleu += imagecolor[ x, y, 0 ]

            barycentre_rouge = int(barycentre_rouge/taille_du_groupe)
            barycentre_vert = int(barycentre_vert/taille_du_groupe)
            barycentre_bleu = int(barycentre_bleu/taille_du_groupe)

            # pour le moment on met les nouveaux cluster au niveau des barycentres
            cluster_bleu[i] = barycentre_bleu
            cluster_vert[i] = barycentre_vert
            cluster_rouge[i] = barycentre_rouge

        # Reaffiliation des points aux nouveaux clusters
        for f in range(K, nb_pixels):
            distance_cluster = numpy.zeros(K)
            y = f % imagecolor.shape[1]
            x = f / imagecolor.shape[1]
            for t in range(0,K):
                distance_cluster[t] += abs(imagecolor[x, y, 2] - cluster_bleu[t])
                distance_cluster[t] += abs(imagecolor[x, y, 1] - cluster_vert[t])
                distance_cluster[t] += abs(imagecolor[x, y, 0] - cluster_rouge[t])
            groupe[f, 0] = numpy.argmin(distance_cluster)

        # Si il n'y a pas de changement par rapport a la derniere etape : on osrt
        if( (old_cluster_bleu==cluster_bleu).all() and (old_cluster_vert==cluster_vert).all() and (old_cluster_rouge==cluster_rouge).all() ):
            there_are_no_modification = 1



    print cluster_bleu
    print cluster_vert
    print cluster_rouge


    #Fin de l'algo, on affiche les resultats

    #On change le format de groupe afin de le rammener au format de l'image d'origine
    groupe=numpy.reshape(groupe, (imagecolor.shape[0], imagecolor.shape[1]))

    #On change chaque pixel de l'image selon le cluster auquel il appartient
    #Il prendre comme nouvelle valeur la position moyenne du cluster
    for i in range(0, imagecolor.shape[0]):
        for j in range(0, imagecolor.shape[1]):
            imagecolor[i,j,0] = (cluster_bleu[int(groupe[i,j])])
            imagecolor[i,j,1] = (cluster_vert[int(groupe[i,j])])
            imagecolor[i,j,2] = (cluster_rouge[int(groupe[i,j])])



	
    # On affiche une fenetre avec l'image
    cv2.namedWindow("sortie")
    #On sort quand l'utilisateur appuie sur une touche
    cv2.imshow("sortie", imagecolor)
    key = cv2.waitKey(0)



if __name__ == "__main__":
    main()