import cv2
import numpy
import random
import time

DISTANCE_DEPLACEMENT = 20
PRINT_IMAGE = 1
DO_NOT_PRINT_IMAGE = 0

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

	return best_seuil

def kmean(path_to_file, param, param_k, print_image = 1 ):

	MAX_LARGEUR = 400
	MAX_HAUTEUR = 400

	K = param_k #Le fameux parametre K de l'algorithme

	start = time.clock()

	somme_etape = 0

	# Charger l'image et la reduire si trop grande (sinon, on risque de passer trop de temps sur le calcul...)
	imagecolor = cv2.imread(path_to_file)
	
	if imagecolor is None:
		print "Le fichier \""+path_to_file+"\" n'as pas ete trouver"
		exit(-1)
	
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

	#afficherImage("Image", imagecolor)

	if (param == "OTSU"):
		imagecolor = cv2.imread(path_to_file)

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
	#for i in range(K,nb_pixels):
	#	groupe[i,0]=random.randint(0, K-1)
	groupe[K:nb_pixels, 0] = numpy.random.randint(0,high=K, size=(nb_pixels-K)).astype('uint8')

	# Initialisation des clusters dans une position aleatoire
	cluster_bleu[0:K] = numpy.random.randint(0, high=255, size=(K)).astype('uint8')
	cluster_vert[0:K] = numpy.random.randint(0, high=255, size=(K)).astype('uint8')
	cluster_rouge[0:K] = numpy.random.randint(0, high=255, size=(K)).astype('uint8')

	#La, c'est a vous d'ecrire le code de la boucle principale
	#Votre code doit faire evoluer les tableaux groupe, cluster_bleu, cluster_rouge et cluster_vert
	#...
	# permet de boucler tant que les clusters changent
	there_are_no_modification = 0

	# Boucle tant que les clusters changent
	while( there_are_no_modification == 0 ):
		somme_etape += 1
		difference = 0

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

			if (taille_du_groupe > 0):
				barycentre_rouge = int(barycentre_rouge/taille_du_groupe)
				barycentre_vert = int(barycentre_vert/taille_du_groupe)
				barycentre_bleu = int(barycentre_bleu/taille_du_groupe)

			# Deplacement des clusters vers vers leurs barycentre.
			if( barycentre_bleu > cluster_bleu[i] ):
				difference += min(DISTANCE_DEPLACEMENT, (barycentre_bleu-cluster_bleu[i]) )
				cluster_bleu[i] = cluster_bleu[i] + min(DISTANCE_DEPLACEMENT, (barycentre_bleu-cluster_bleu[i]) )
			else:
				difference += min(DISTANCE_DEPLACEMENT, (cluster_bleu[i]-barycentre_bleu))
				cluster_bleu[i] = cluster_bleu[i] - min(DISTANCE_DEPLACEMENT, (cluster_bleu[i]-barycentre_bleu))
			if( barycentre_vert > cluster_vert[i] ):
				difference += min(DISTANCE_DEPLACEMENT, (barycentre_vert-cluster_vert[i]) )
				cluster_vert[i] = cluster_vert[i] + min(DISTANCE_DEPLACEMENT, (barycentre_vert-cluster_vert[i]) )
			else:
				difference += min(DISTANCE_DEPLACEMENT, (cluster_vert[i]-barycentre_vert))
				cluster_vert[i] = cluster_vert[i] - min(DISTANCE_DEPLACEMENT, (cluster_vert[i]-barycentre_vert))
			if( barycentre_rouge > cluster_rouge[i] ):
				difference += min(DISTANCE_DEPLACEMENT, (barycentre_rouge-cluster_rouge[i]) )
				cluster_rouge[i] = cluster_rouge[i] + min(DISTANCE_DEPLACEMENT, (barycentre_rouge-cluster_rouge[i]) )
			else:
				difference += min(DISTANCE_DEPLACEMENT, (cluster_rouge[i]-barycentre_rouge))
				cluster_rouge[i] = cluster_rouge[i] - min(DISTANCE_DEPLACEMENT, (cluster_rouge[i]-barycentre_rouge))

		# Reaffiliation des points aux nouveaux clusters		
		distances_cluster2 = (bleu[:, numpy.newaxis] - cluster_bleu)**2 + (rouge[:, numpy.newaxis] - cluster_rouge)**2 + (vert[:, numpy.newaxis] - cluster_vert)**2
		groupe = numpy.argmin(distances_cluster2, axis=2)


		# Si il n'y a pas de changement par rapport a la derniere etape : on sort
		if( difference == 0 ):
			there_are_no_modification = 1

	# Fin de l'algo, on affiche les resultats

	# On change le format de groupe afin de le rammener au format de l'image d'origine
	groupe=numpy.reshape(groupe, (imagecolorRes.shape[0], imagecolorRes.shape[1]))
	
	# si on affiche l'image, alors on convertis sinon perte de temps
	if print_image:
		# On change chaque pixel de l'image selon le cluster auquel il appartient
		# Il prendre comme nouvelle valeur la position moyenne du cluster
		for i in xrange(0, imagecolorRes.shape[0]):
			for j in xrange(0, imagecolorRes.shape[1]):
				imagecolorRes[i,j,0] = (cluster_bleu[int(groupe[i,j])])
				imagecolorRes[i,j,1] = (cluster_vert[int(groupe[i,j])])
				imagecolorRes[i,j,2] = (cluster_rouge[int(groupe[i,j])])

	if (param == "HSV"):
		imagecolorRes = cv2.cvtColor(imagecolorRes, cv2.COLOR_HSV2BGR)
	elif(param == "LAB"):
		imagecolorRes = cv2.cvtColor(imagecolorRes, cv2.COLOR_LAB2BGR)
	#elif(param == "OTSU"): ### rajouter un seuil
	#	imagecolorRes = seuil(imagecolorRes, best_seuil)

	if print_image :
		afficherImage("sortie", imagecolorRes)
	
	return sorted(zip(cluster_bleu, cluster_vert, cluster_rouge))

def find_nb_cluster(file_to_path, param):
	number_test = 20
	
	print 'filename : %s, number_test : %s' % (file_to_path, number_test)
	
	for k_test in xrange(2, 15):
		t = []
		for i in xrange(0, number_test):
			t.append(kmean(file_to_path, param, k_test, DO_NOT_PRINT_IMAGE))
		
		tot = 0
		# on trouve les point associe d'un cluster a un autre et on calcul la somme des distances.
		for l in xrange(1, number_test):
			for i in xrange(0, k_test):
				d = 1000000
				
				for j in xrange(0, k_test):
				
					d_temp = ((t[0][i][0] - t[l][j][0])**2 + (t[0][i][1] - t[l][j][1])**2 + (t[0][i][2] - t[l][j][2])**2)**0.5
					if d_temp < d:
						d = d_temp
						
				tot +=d
		
		print 'nb cluster : %s, distance = %s' % (k_test, tot)
	
if __name__ == "__main__":
	#numpy.random.seed( 0 )

	# 1
	#kmean('perr.jpg', "normal", 32)
	
	# 2
	#kmean('perr.jpg', "normal", 32)
	#kmean('perr.jpg', "HSV", 32)
	#kmean('perr.jpg', "LAB", 32)
	
	# 3
	#kmean('ville.png', "OTSU", 2)
	
	# 4
	find_nb_cluster('mimi2.jpg', "normal")