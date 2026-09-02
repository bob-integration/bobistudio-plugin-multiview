# Multiview

Compose N sources vidéo en une mosaïque et produit un shm de sortie câblable vers un encodeur ou un sender. Layout configurable via l'éditeur drag-and-drop.

## Configurer le layout

Depuis **Traitements → Multiview** : ouvrir l'éditeur de layout pour positionner les fenêtres, choisir les sources, régler la résolution de sortie.

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

## Tally et libellés

Deux modes, et **le mode central est celui par défaut**.

### Mode central (défaut)

C'est l'orchestrateur qui pousse au mur, en direct, la couleur de tally **et** le texte de chaque
fenêtre. Le mur n'a aucun serveur à lui, ne parle aucun protocole, et n'a pas besoin d'être
redéployé quand un libellé change.

**Le tally s'adresse PAR SOURCE.** Une fenêtre reçoit l'état du flux qu'elle affiche, sur le
**niveau** que vous lui avez donné. Un niveau est une entité nommée — « Antenne », « Plateau » —
qui appartient à une production ; ce n'est pas un numéro de trame, et changer de protocole demain
ne changerait rien à ce réglage. Une fenêtre sans niveau n'a pas de tally, ce qui est le cas
courant : un instrument n'est pas à l'antenne.

> ⚠ Le rouge et le vert d'une fenêtre viennent du **même niveau**. Une source à la fois au
> programme et en préparation ne consomme donc pas deux entrées : les deux états se cumulent sur
> son niveau et donnent l'**ambre**, qui allume les deux bandeaux. C'est ainsi que l'orange se voit
> sur le mur.

**Le libellé vient du tableau des sources** (page Libellés), organisé en colonnes. Chaque fenêtre
choisit la colonne qu'elle affiche. Les colonnes sont poussées **en direct** : éditer un libellé
se voit sur le mur sans redéploiement.

> ⚠ **Une colonne vide n'hérite jamais du texte précédent.** Si la colonne demandée est vide pour
> cette source, l'orchestrateur remonte les autres colonnes pour en trouver une remplie ; s'il n'en
> trouve aucune, la fenêtre retombe sur son propre nom. Elle ne garde JAMAIS le libellé de la
> source qu'elle affichait avant — c'est un défaut qui a existé, et qui faisait afficher le nom
> d'un enregistreur sur le mélangeur.

Une fenêtre dont la source n'a ni tally ni libellé s'**éteint** et affiche son nom. Elle ne reste
pas figée sur son état d'avant : ne pas savoir n'autorise pas à laisser croire.

### Mode direct

Le mur ouvre son propre serveur **TSL 5.0** (TCP, port 4801) et reçoit le tally d'un contrôleur
sans passer par l'orchestrateur. Réservé aux cas où l'on veut court-circuiter le contrôleur
central. Dans ce mode, l'orchestrateur ne pousse rien — il ne faut pas piloter des deux côtés.

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

## Les frises : que s'est-il passé sur cette source ?

Un mur montre l'instant présent. Une **frise** montre la minute écoulée — c'est l'outil à ouvrir
quand quelqu'un dit « il y a eu quelque chose il y a trente secondes ».

**Frise vidéo** — une vignette par seconde, et sous la bande un ruban d'événements : **gel**,
**noir**, **perte de signal**.

> ★ La vignette est capturée **à l'instant de l'événement**, puis épinglée dans sa case : la bande
> montre l'image **sur laquelle ça s'est figé**, et pour une perte de signal la dernière image
> valide — pas une image quelconque prise dans la seconde.

**Frise audio** — enveloppe des crêtes, **saturation** persistante en rouge, **silence** grisé. Un
canal muet depuis quarante secondes se voit d'un coup d'œil.

**Profondeur** : 10, 30, 60 ou 120 secondes.

**Deux façons de la poser**, avec le même rendu :

| | source | géométrie |
|---|---|---|
| Composant d'un modèle de fenêtre | celle de la fenêtre | relative à la cellule |
| Bloc libre du mur | câblée en propre | relative au mur entier |

Le coût pour la trame est négligeable : la recomposition d'une frise, qui prend des dizaines de
millisecondes, est faite par un fil dédié et non dans la boucle de mixage. En ajouter une ne fait
donc pas tomber la cadence — c'est le **câblage** de sa source qu'il faut prévoir (un bloc de mur
a besoin de sa propre entrée vidéo ou audio).

## Réglages avancés (panneau ⚙ du conteneur)

| Réglage | Effet |
|---|---|
| Orientation | Paysage (défaut), ou portrait (rotation horaire/anti-horaire) — pour un mur destiné à un affichage vertical |
| Filtre de réduction des vignettes | **Moyenne du bloc** (défaut) : anticrénelage correct, le texte incrusté et les détails fins de la source restent lisibles réduits ; coûte environ 10× plus de CPU par vignette qu'une simple **décimation** (comportement historique, plus rapide mais crénelé). Si le nœud est chargé, surveiller `compose_breakdown_ms.inputs` sur `:8080` avant de garder la moyenne sur beaucoup de tuiles |
| Sources entrelacées | **Tissage** (défaut) : recompose les deux champs, résolution verticale complète — nécessaire pour que le texte incrusté d'une source 1080i reste lisible. Peut peigner sur du mouvement rapide, largement absorbé par la réduction en vignette. **Champ seul** (historique) : moitié de la résolution verticale, mais insensible au mouvement |
| Mode tranche | Lit les entrées par tranches et publie la sortie au fur et à mesure, au lieu d'attendre l'image entière. **Un étage en image entière coûte une trame de latence à toute la chaîne qui le traverse**, et cette dette n'apparaît sur aucun compteur — le mur affiche une cadence parfaite. Incompatible avec le portrait et l'entrelacé, qui restent en image entière |
| Tranche GPU | Le mode tranche sur carte. Opt-in : à n'activer qu'après validation sur l'installation, la carte et le format visés |
| Afficher « NO SIGNAL » | Bandeau quand une fenêtre n'a pas de source ou pas de grain |
| Détection image figée (s) | Délai avant d'afficher l'alerte figé sur une fenêtre, 0 = désactivé |

## Latence : un mur coûte une image entière, ou rien

Un étage de la chaîne ne coûte pas son temps de calcul — il coûte **une image entière dès qu'il
fait rater le train**, et rien du tout sinon. C'est un escalier, pas une pente.

**Le mécanisme.** L'émetteur 2110 vient chercher le contenu à un instant fixe du créneau : environ
**16,4 ms** après son début, sur un moteur à 50 images/s (mesuré le 2026-08-12). Tout ce qui est
publié avant part à l'image suivante ; tout ce qui arrive après attend une image de plus. La
chaîne dispose donc d'un **budget d'environ 16 ms par image**, que chaque étage consomme.

**La règle par mur**, à partir de deux chiffres publiés par le conteneur sur `:8080` :

> arrivée de l'entrée (`inputs_latency_ms`) + temps propre (`own_latency_ms`) < ~16 ms

Sous le seuil, le mur est *gratuit* pour la latence de sortie. Au-dessus, il coûte une image
pleine — et un mur très chargé peut le franchir deux fois.

**Deux configurations mesurées, pour situer :**

| mur | temps propre | arrivée entrée | cumul | coût |
|---|---|---|---|---|
| 1 tuile, CPU, sans habillage | 2,3 ms | 14,3 ms | 16,6 ms | 1 image |
| 4 tuiles, GPU, libellés + horloge + bandeau | 11,2 ms | 9,5 ms | 20,7 ms | **2 images** |

Le second travaille cinq fois plus longtemps et coûte une image de plus. C'est ce qui rend la
question « combien coûte un multiview ? » sans réponse générale : **cela dépend de la
configuration**, et seul le cumul ci-dessus le dit.

**Ce qui NE coûte rien**, mesuré sur la même chaîne : les proxies de la pyramide, la réplication
RDMA entre nœuds, et un étage de traitement simple (un correcteur de couleur : 2 ms, et deux en
cascade restent additifs). Tous restent sous le budget.

**Où gagner, si un mur dépasse.** Le temps propre est le seul poste sur lequel agir — l'arrivée
des entrées dépend des étages amont. Les leviers habituels : moins de tuiles, moins d'habillage
(chaque VU-mètre, horloge et libellé s'ajoute), le filtre de réduction en décimation plutôt qu'en
moyenne de bloc, et le mode tranche. Le détail par poste est dans `compose_breakdown_ms` sur
`:8080`.

⚠ **Deux pièges de lecture.**

- Le seuil de 16 ms est celui de **notre émetteur 2110**. Un aval différent — un mur en cascade,
  un enregistreur, un encodeur — a le sien, qui reste à mesurer.
- `own_latency_ms` est une **moyenne**. Un mur qui franchit le seuil par intermittence alterne
  entre deux valeurs de latence sans que la moyenne le montre ; `delai_etage_trames` (bornes
  `recent` et `vieux`) et le compteur de trames lentes le trahissent mieux.

⚠ **La cadence (`flow` / `genlock`) ne change pas la latence** : mesuré, les deux donnent le même
délai de bout en bout. `flow` aligne les index entre étages, ce dont a besoin un mur **shardé**
pour que son assembleur retrouve tous ses morceaux au même index — pas la latence.

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
