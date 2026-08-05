# Multiviewer

Compose N sources vidéo en une mosaïque et produit un shm de sortie câblable vers un encodeur ou un sender. Layout configurable via l'éditeur drag-and-drop.

## Configurer le layout

Depuis **Traitements → Multiviewer** : ouvrir l'éditeur de layout pour positionner les fenêtres, choisir les sources, régler la résolution de sortie.

Les modifications sont **live** (pas de redéploiement). Le layout est persisté automatiquement.

## Câblage

Câbler les sources depuis la page **Câbles** ou depuis l'éditeur directement. Le changement de source se fait **à chaud** si la résolution est identique.

## Proxies pyramide (accélération)

Le multiview peut lire des versions **pré-réduites** des sources, produites par une **pyramide de
proxies**, au lieu de redimensionner la source pleine pour chaque tuile — gain important sur le
temps de traitement (le 1080p50 tient même avec beaucoup de tuiles).

C'est **opportuniste et automatique** : il suffit qu'une *pyramide* soit déployée **sur le même
nœud** et câblée aux **mêmes sources** que les fenêtres du multiview. On ne câble **pas** le
multiview vers la pyramide ; les deux consomment la même source. Après avoir câblé la pyramide,
**re-déploie/re-sauve le multiview** pour qu'il prenne les proxies en compte. Sans pyramide, le
multiview lit la source pleine comme avant (aucune régression).

Pour voir ce que chaque tuile lit, active **« Proxies pyramide (ingénierie) »** dans les réglages
globaux du composer : un badge apparaît en haut-gauche de chaque vignette — `¼ ~` (octave),
`952×536 ✓` (taille sur-mesure, copie pure) ou `plein ↯` (source pleine, pas encore de proxy). La
bascule est **à chaud**. Détail complet dans l'aide du plugin **Pyramide**.

## Modèles de PiP

L'habillage d'une fenêtre (nom/UMD, tally, VU-mètres, métadonnées ANC, horloge, texte, badge
format) peut être remplacé par un **modèle de PiP** composé librement dans **Réglages → PiP** :
chaque composant se place et se dimensionne par drag & drop, en coordonnées relatives — un même
modèle sert à une tuile plein écran comme à une vignette. Les composants peuvent être
**conditionnels** (visibles seulement sur tally rouge/vert, perte de signal, image figée…) et
porter un seuil « masquer sous N px » pour rester lisibles sur les petites tuiles.

L'affectation se fait fenêtre par fenêtre via le champ **Modèle de PiP** du panneau d'entrée
(application à chaud, sans coupure). « Habillage classique » = comportement historique,
strictement inchangé. Trois modèles d'usine (Production, Ingénierie, Minimal) servent de point
de départ ; dupliquez-les pour les adapter.

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
  **PTP** (horloge synchrone du jour, avec offset positif/négatif), **chrono**, **décompte**, ou
  **ANC** (timecode embarqué RP188/ATC de la source — choisir l'entrée vidéo dont on lit le TC ;
  affiche `--:--:--:--` si la source n'a pas de timecode). Pour chrono/décompte, des boutons
  **Démarrer / Arrêter / Réinitialiser** pilotent l'horloge en direct.
- **Champ image** : ajoute un **logo** au premier plan, ou un **fond** derrière toute l'image.
  L'image est **importée directement depuis l'ordinateur** (bouton « Importer une image… »),
  réduite côté navigateur puis embarquée au déploiement (aucun stockage serveur). Ajustement
  `contain` / `cover` / `stretch` et opacité réglables.

> La source **ANC** lit le timecode embarqué (ST 2110-40 / RP188) du **flux ANC** de la source,
> décodé côté multiview. Elle nécessite une source reçue par le moteur MTL (2110_io) qui produit
> le flux ANC ; les sources générateur/ffmpeg sans ANC affichent `--:--:--:--`.

## Réglages avancés (panneau ⚙ du conteneur)

| Réglage | Effet |
|---|---|
| Orientation | Paysage (défaut), ou portrait (rotation horaire/anti-horaire) — pour un mur destiné à un affichage vertical |
| Filtre de réduction des vignettes | **Moyenne du bloc** (défaut) : anticrénelage correct, le texte incrusté et les détails fins de la source restent lisibles réduits ; coûte environ 10× plus de CPU par vignette qu'une simple **décimation** (comportement historique, plus rapide mais crénelé). Si le nœud est chargé, surveiller `compose_breakdown_ms.inputs` sur `:8080` avant de garder la moyenne sur beaucoup de tuiles |
| Sources entrelacées | **Tissage** (défaut) : recompose les deux champs, résolution verticale complète — nécessaire pour que le texte incrusté d'une source 1080i reste lisible. Peut peigner sur du mouvement rapide, largement absorbé par la réduction en vignette. **Champ seul** (historique) : moitié de la résolution verticale, mais insensible au mouvement |
| Afficher « NO SIGNAL » | Bandeau quand une fenêtre n'a pas de source ou pas de grain |
| Détection image figée (s) | Délai avant d'afficher l'alerte figé sur une fenêtre, 0 = désactivé |

## Câblage — au-delà de la vidéo

Une fenêtre peut aussi recevoir, câblés séparément et optionnels : un flux **audio** (pour ses
VU-mètres), un flux **ANC** (pour la source d'horloge ANC d'un overlay horloge) et, pour
l'historique vidéo/audio, des blocs dédiés. Le multiview n'exige que la vidéo — les absences ne
bloquent rien, elles désactivent seulement les fonctions associées (pas de VU sans audio câblé,
pas de TC ANC sans flux ANC câblé).

## Notes

- La résolution de sortie est fixée au déploiement
- Les fenêtres dont la source a une résolution différente sont redimensionnées automatiquement
- Le bouton Monitoring prévisualise la sortie du multiviewer
