# Multiviewer

Compose N sources vidéo en une mosaïque et produit un shm de sortie câblable vers un encodeur ou un sender. Layout configurable via l'éditeur drag-and-drop.

## Configurer le layout

Depuis **Traitements → Multiviewer** : ouvrir l'éditeur de layout pour positionner les fenêtres, choisir les sources, régler la résolution de sortie.

Les modifications sont **live** (pas de redéploiement). Le layout est persisté automatiquement.

## Câblage

Câbler les sources depuis la page **Câbles** ou depuis l'éditeur directement. Le changement de source se fait **à chaud** si la résolution est identique.

## Layouts enregistrés

Enregistrer une disposition (bouton « Enregistrer le layout ») pour la rappeler instantanément plus tard. Les layouts sont globaux (partagés entre instances).

## Tally

Le multiviewer expose un endpoint TSL 5.0 (TCP, port 4801) pour signalisation tally rouge/vert sur les fenêtres.

## Notes

- La résolution de sortie est fixée au déploiement
- Les fenêtres dont la source a une résolution différente sont redimensionnées automatiquement
- Le bouton Monitoring prévisualise la sortie du multiviewer
