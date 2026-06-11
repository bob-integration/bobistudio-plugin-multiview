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

## Outils d'overlay (texte / horloge / image)

En plus des fenêtres vidéo, on peut poser des **objets d'overlay** sur le composer (barre
« Outils »). Ce sont des objets purement visuels (non câblés) déplaçables/redimensionnables
comme les fenêtres. Toute modification s'applique **à chaud**.

- **Champ texte** : affiche du texte (sans vidéo). Réglages : police, couleur du texte,
  couleur de fond. Chaque couleur a un état alternatif **Tally On** (autre couleur) déclenché
  par un index TSL. Le texte est **saisi en local** ou **issu du TSL** (index TSL).
- **Champ horloge** : affiche un timecode réglable (cases **HH / MM / SS / II**). Source :
  **PTP** (horloge synchrone du jour, avec offset positif/négatif), **chrono** ou **décompte**.
  Pour chrono/décompte, des boutons **Démarrer / Arrêter / Réinitialiser** pilotent l'horloge
  en direct.
- **Champ image** : ajoute un **logo** au premier plan, ou un **fond** derrière toute l'image.
  L'image est **importée directement depuis l'ordinateur** (bouton « Importer une image… »),
  réduite côté navigateur puis embarquée au déploiement (aucun stockage serveur). Ajustement
  `contain` / `cover` / `stretch` et opacité réglables.

> La source horloge « entrée ANC depuis SHM » est prévue dans une version ultérieure (elle
> nécessite l'extraction du timecode en amont, côté receiver).

## Notes

- La résolution de sortie est fixée au déploiement
- Les fenêtres dont la source a une résolution différente sont redimensionnées automatiquement
- Le bouton Monitoring prévisualise la sortie du multiviewer
