Adaptation de l'architecture LeNet pour classifier des panneaux de signalisations de la base GTSRB (http://benchmark.ini.rub.de/)

Utilisation de la bibliothèque Keras, sur-couche à Theano

L'implantation initiale de LeNet provient de cet article : http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/

schéma de l'architecture : ![architecture](./arch.png)

# A Faire

## Script de chargement des données
- [*] charger les données en mémoire
- [*] modifier l'encodage des images (triplet RGB -> int RGB)
- [ ] redimensionner les images pour qu'elles soient compatibles avec le champ réceptif de LeNet
- [ ] scale the data to the range [0, 1.0] ?????
- [*] séparer les données en ensemble d'entraînement et ensemble de validation
- [ ] enregistrer les données sur le disque pour éviter le pré-traitement

## Performance du script de chargement
- [ ] conversion des images (triplet -> int) 

## Adaptation de LeNet
- [ ] Adapter le champ réceptif de LeNet pour qu'il gère couleurs


