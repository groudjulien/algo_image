import cv2
import numpy
import random
import time

# Fonction qui permet d'afficher une image dans une fenetre
# @param {STRING} nomFenetre - nom de la fenetre qui sera cree
# @param {MATRICE} img - une image a afficher
# @return void - ne retourne rien
def afficherImage( nomFenetre , img ):
    # Affichage de l'image
    cv2.imshow(nomFenetre, img)

    # On attend que l'utilisateur appuie sur une touche => temps infinie
    key = cv2.waitKey(0)

    # on ferme toutes les fenetre ouverte
    cv2.destroyAllWindows()

# Fonction qui realise un seuil
# @param {MATRICE} img - image source
# @param {MATRICE} valeur_seuil - la valeur de seuil
# @return {MATRICE} - retourne l'image seuille
def seuil( img, valeur_seuil):
    r = numpy.zeros(img.shape, img.dtype)
    idx = (img>=valeur_seuil)
    r[idx] = numpy.iinfo(img.dtype).max
    return r

def seuil_otsu( histo ):

    # calcul de la moyenne des pixels
    somme = 0
    somme_positif = 0
    nb_pixel_positif = 0
    val_max_pixel = 0
    for i in xrange(0,256):
        if (i > 0 and histo[i] > 0):
            somme_positif = somme_positif + histo[i]*i
            nb_pixel_positif = nb_pixel_positif + histo[i]
        if(i > val_max_pixel and histo[i] > 0):
            val_max_pixel = i

    N = numpy.sum(histo)

    # calcul des premiers parametrages
    g1 = histo[0]
    mu1 = 0
    g2 = N - g1
    mu2 = float(somme_positif / nb_pixel_positif)

    best_var = (g1 * g2) / (N * N) * ((mu1 - mu2) * (mu1 - mu2))
    best_seuil = 0

    #print N
    #print g2
    #print mu2
    #print best_var
    #print val_max_pixel

    for t in xrange(1,val_max_pixel):
        mu1 = mu1 * g1 + t * histo[t]
        mu2 = mu2 * g2 - t * histo[t]

        g1 = g1 + histo[t]
        g2 = g2 - histo[t]

        if g1 > 0:
            mu1 = mu1 / g1

        if g2 > 0:
            mu2 = mu2 / g2

        if g1 > 0 and g2 > 0:
            if (g1*g2) / (N*N)*((mu1-mu2)*(mu1-mu2)) > best_var:
                best_var = (g1*g2) / (N*N)*((mu1-mu2)*(mu1-mu2))
                best_seuil = t

    #print best_seuil
    #print best_var

    return best_seuil

def main( param, param_k ):
    MAX_LARGEUR = 400
    MAX_HAUTEUR = 400

    K = param_k #Le fameux parametre K de l'algorithme

    start = time.clock()

    somme_etape = 0

    # Charger l'image et la reduire si trop grande (sinon, on risque de passer trop de temps sur le calcul...)
    if( param == "OTSU" ):
        imagecolor = cv2.imread('ville.png')
    else:
        imagecolor = cv2.imread('perr.jpg')
    if imagecolor.shape[0] > MAX_LARGEUR or imagecolor.shape[1] > MAX_HAUTEUR:
        factor1 = float(MAX_LARGEUR) / imagecolor.shape[0]
        factor2 = float(MAX_HAUTEUR) / imagecolor.shape[1]
        factor = min(factor1, factor2)
        imagecolor = cv2.resize(imagecolor, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

    imagecolorRes = numpy.copy(imagecolor)

    if (param == "HSV"):
        imagecolorRes = cv2.cvtColor(imagecolorRes, cv2.COLOR_BGR2HSV)
    elif (param == "LAB"):
        imagecolorRes = cv2.cvtColor(imagecolorRes, cv2.COLOR_BGR2LAB)

    # Le nombre de pixels de l'image
    nb_pixels = imagecolorRes.shape[0] * imagecolorRes.shape[1]

    afficherImage("Image", imagecolor)

    if (param == "OTSU"):
        imagecolor = cv2.imread('ville.png')

        imagecolorOtsu = numpy.copy(imagecolor)

        histo = cv2.calcHist([imagecolorOtsu], [0], None, [256], [0, 256])
        best_seuil = seuil_otsu(histo)
        imagecolorOtsu = seuil(imagecolorOtsu, best_seuil)

        afficherImage("Otsu", imagecolorOtsu)


    #Les coordonnees BRV de tous les pixels de l'image (les elements de E)
    bleu = imagecolorRes[:, :, 0].reshape(nb_pixels, 1)
    vert = imagecolorRes[:, :, 1].reshape(nb_pixels, 1)
    rouge = imagecolorRes[:, :, 2].reshape(nb_pixels, 1)




    #Les coordonnees BRV de chaque point-cluster (les elements de N)
    cluster_bleu = numpy.zeros(K)
    cluster_vert = numpy.zeros(K)
    cluster_rouge = numpy.zeros(K)


    #Ce tableau permet de connaitre, pour chaque pixel de l'image, a quel cluster il appartient
    #On le remplit au hasard
    groupe = numpy.zeros((nb_pixels, 1)) #groupe est un tableau de Card(E) cases, et chaque valeur est un entier entre 0 et K-1, designant le cluster auquel chaque point sera rattache
    #On remplit au hasard le tableau groupe, c'est a dire que l'on attribue au hasard chaque point de l'espace a un des K clusters
    #Cependant, pour etre sur qu'au depart chaque cluster est rattache a au moins un point de l'espace, on attribue les K premiers points de l'espace a chaque K clusters
    for i in xrange(0,K):
        groupe[i,0]=i
    #La, on fait l'attribution du reste des points de l'espace a des clusters choisis au hasard
    for i in xrange(K,nb_pixels):
        groupe[i,0]=random.randint(0, K-1)

    # Initialisation des clusters dans une position aleatoire
    for i in xrange(0, K):
        cluster_bleu[i]=random.randint(0, 255)
        cluster_vert[i]=random.randint(0, 255)
        cluster_rouge[i]=random.randint(0, 255)

    #La, c'est a vous d'ecrire le code de la boucle principale
    #Votre code doit faire evoluer les tableaux groupe, cluster_bleu, cluster_rouge et cluster_vert
    #...
    # permet de boucler tant que les clusters changent
    there_are_no_modification = 0

    #print "Nombre de linges et colonnes"
    #print imagecolorRes.shape[0]
    #print imagecolorRes.shape[1]

    # Boucle tant que les clusters changent
    while( there_are_no_modification == 0 ):

        somme_etape += 1

        old_cluster_bleu = numpy.copy(cluster_bleu)
        old_cluster_vert = numpy.copy(cluster_vert)
        old_cluster_rouge = numpy.copy(cluster_rouge)

        # Calcul des positions des nouveaux points cluster
        for i in xrange(0,K):
            barycentre_rouge = 0
            barycentre_vert = 0
            barycentre_bleu = 0

            taille_du_groupe = 0

            # Calcul du barycentre
            idx = (groupe[:,0] == i)
            taille_du_groupe = (groupe[:,0] == i).sum()
            barycentre_rouge = numpy.sum(rouge[idx])
            barycentre_vert = numpy.sum(vert[idx])
            barycentre_bleu = numpy.sum(bleu[idx])

            #for j in xrange( 0, nb_pixels-1 ):
            #    if( groupe[j,0] == i ):
            #        barycentre_rouge = barycentre_rouge + rouge[j,0]
            #        barycentre_vert = barycentre_vert + vert[j,0]
            #        barycentre_bleu = barycentre_bleu + bleu[j,0]
            #        taille_du_groupe += 1

            #print "Groupe = "+str(i)+" de taille = "+str(taille_du_groupe)

            #print "Barycentre rouge : "+str(barycentre_rouge)
            #print "Barycentre vert : "+str(barycentre_vert)
            #print "Barycentre bleu : "+str(barycentre_bleu)

            barycentre_rouge = int(barycentre_rouge/taille_du_groupe)
            barycentre_vert = int(barycentre_vert/taille_du_groupe)
            barycentre_bleu = int(barycentre_bleu/taille_du_groupe)

            #print "Barycentre rouge : "+str(barycentre_rouge)
            #print "Barycentre vert : "+str(barycentre_vert)
            #print "Barycentre bleu : "+str(barycentre_bleu)

            # pour le moment on met les nouveaux cluster au niveau des barycentres
            if( barycentre_bleu > cluster_bleu[i] ):
                cluster_bleu[i] = cluster_bleu[i] + min(10, (barycentre_bleu-cluster_bleu[i]) )
            else:
                cluster_bleu[i] = cluster_bleu[i] - min(10, (cluster_bleu[i]-barycentre_bleu))
            if( barycentre_vert > cluster_vert[i] ):
                cluster_vert[i] = cluster_vert[i] + min(10, (barycentre_vert-cluster_vert[i]) )
            else:
                cluster_vert[i] = cluster_vert[i] - min(10, (cluster_vert[i]-barycentre_vert))
            if( barycentre_rouge > cluster_rouge[i] ):
                cluster_rouge[i] = cluster_rouge[i] + min(10, (barycentre_rouge-cluster_rouge[i]) )
            else:
                cluster_rouge[i] = cluster_rouge[i] - min(10, (cluster_rouge[i]-barycentre_rouge))

            #print "cluster_rouge : "+str(cluster_rouge[i])
            #print "cluster_vert : "+str(cluster_vert[i])
            #print "cluster_bleu : "+str(cluster_bleu[i])

        #print cluster_rouge
        #print cluster_vert
        #print cluster_bleu

        # Reaffiliation des points aux nouveaux clusters
        valeurs_connus = dict()
        for f in xrange(K, nb_pixels):
            index_k_min = -1
            if ( valeurs_connus.has_key(bleu[f,0]) and valeurs_connus[bleu[f,0]].has_key(vert[f,0]) and valeurs_connus[bleu[f,0]][vert[f,0]].has_key(rouge[f,0]) ):
                index_k_min = valeurs_connus[bleu[f,0]][vert[f,0]][rouge[f,0]]
            else :
                distance_cluster = numpy.zeros(K)
                distance_reference = -1
                for t in xrange(0, K):
                    distance_cluster[t] = ((bleu[f,0] - cluster_bleu[t])) ** 2 + abs(
                        (vert[f,0] - cluster_vert[t])) ** 2 + abs((rouge[f,0] - cluster_rouge[t])) ** 2
                    if (index_k_min == -1 or distance_cluster[t] < distance_reference):
                        distance_reference = distance_cluster[t]
                        index_k_min = t
                        if( not valeurs_connus.has_key(bleu[f,0]) ):
                            valeurs_connus[bleu[f, 0]] = dict()
                        if( not valeurs_connus[bleu[f,0]].has_key(vert[f,0]) ):
                            valeurs_connus[bleu[f, 0]][vert[f, 0]] = dict()
                        valeurs_connus[bleu[f,0]][vert[f,0]][rouge[f,0]] = t

            groupe[f, 0] = index_k_min

        # Si il n'y a pas de changement par rapport a la derniere etape : on osrt
        if( (old_cluster_bleu==cluster_bleu).all() and (old_cluster_vert==cluster_vert).all() and (old_cluster_rouge==cluster_rouge).all() ):
            there_are_no_modification = 1



    print cluster_bleu
    print cluster_vert
    print cluster_rouge


    #Fin de l'algo, on affiche les resultats

    #On change le format de groupe afin de le rammener au format de l'image d'origine
    groupe=numpy.reshape(groupe, (imagecolorRes.shape[0], imagecolorRes.shape[1]))

    #On change chaque pixel de l'image selon le cluster auquel il appartient
    #Il prendre comme nouvelle valeur la position moyenne du cluster
    for i in xrange(0, imagecolorRes.shape[0]):
        for j in xrange(0, imagecolorRes.shape[1]):
            imagecolorRes[i,j,0] = (cluster_bleu[int(groupe[i,j])])
            imagecolorRes[i,j,1] = (cluster_vert[int(groupe[i,j])])
            imagecolorRes[i,j,2] = (cluster_rouge[int(groupe[i,j])])

    if (param == "HSV"):
        imagecolorRes = cv2.cvtColor(imagecolorRes, cv2.COLOR_HSV2BGR)
    elif(param == "LAB"):
        imagecolorRes = cv2.cvtColor(imagecolorRes, cv2.COLOR_LAB2BGR)

    end = time.clock()
    value = end - start
    print "Duree = "+str(value)+"secondes pour K="+str(K)+", pour "+str(somme_etape)+" etapes et pour le mode "+str(param)

    afficherImage("sortie", imagecolorRes)



if __name__ == "__main__":
    main( "normal",3 )
    main( "HSV",3 )
    main( "LAB",3 )
    #main( "OTSU",2 )