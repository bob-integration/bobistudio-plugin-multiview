# Multiview

*[English version](README.en.md)*

Composite plusieurs sources vidéo en un seul flux de sortie. Il fait partie de
[Bobi.Studio](https://github.com/bob-integration/bobistudio), un orchestrateur broadcast bâti
sur le bus ST 2110 / MXL.

---

## Ce qu'il fait

Chaque fenêtre lit un flux du bus MXL, le réduit et le place. Autour de l'image viennent les
éléments d'un mur de production : bandeau de nom, lampes de tally, vumètres audio, cadre de
viseur, horloges, décomptes, incrustations de texte.

La disposition est décrite par un **modèle** — une liste de composants positionnés en
coordonnées relatives — que les fenêtres héritent du mur ou redéfinissent une à une. Rien n'est
câblé en dur : un mur régulier, une mosaïque asymétrique et un PiP plein écran sont le même code
avec des modèles différents.

**Combien de fenêtres ?** La question se pose mal, et c'est utile de savoir pourquoi. Quarante
fenêtres ont déjà tourné sur un seul mur — mais nues : sans bandeau, sans vumètre, sans frise.
Le coût est dans ce qui ENTOURE l'image, pas dans le nombre d'images : chaque élément d'habillage
a son propre prix, et c'est leur somme qui décide, avec le nœud qui porte le mur. Un chiffre
seul induirait en erreur dans les deux sens. Le mur publie donc son budget de trame et la
ventilation de son temps de composition, pour qu'on décide sur la mesure de SON installation.

---

## Les frises : « que s'est-il passé sur cette source ? »

C'est l'outil le moins courant de ce mur, et celui qui change le plus la façon de l'exploiter.
Un multiviewer montre l'instant présent ; une frise montre la **minute écoulée**.

**Frise vidéo** — une vignette par seconde, et sous la bande un ruban d'événements : gel, noir,
perte de signal. L'échantillonnage tourne dans son propre fil, jamais dans la boucle de mixage.

> ★ **La vignette est capturée À L'INSTANT de l'événement**, puis épinglée dans sa case
> temporelle : l'échantillonnage régulier ne l'écrase plus. La bande montre donc l'image **sur
> laquelle ça s'est figé** — et pour une perte de signal, la dernière image valide — au lieu
> d'une image quelconque prise dans la seconde. C'est ce détail qui fait la différence entre
> « il y a eu un incident » et « voici ce qu'on diffusait quand il est arrivé ».

**Frise audio** — enveloppe des crêtes, saturation persistante en rouge, silence grisé. Un canal
muet depuis quarante secondes se voit d'un coup d'œil, sans avoir eu à le surveiller.

Profondeur au choix : 10, 30, 60 ou 120 secondes. Chaque frise existe sous deux formes, avec le
même code de rendu — composant d'un modèle de fenêtre (la source est celle de la fenêtre), ou
bloc libre posé sur le mur avec sa source propre.

**Ce que ça coûte à la trame : presque rien**, et c'est un choix de conception. Recomposer une
frise coûte des dizaines de millisecondes ; ce travail est fait par un fil dédié qui publie des
tuiles prêtes, et la boucle de mixage ne paie que leur incrustation. Un étage qui recomposerait
dans la trame perdrait l'image.

---

## L'éditeur

Le mur se dessine dans le navigateur, sur un aperçu de ce que verra l'écran — pas sur une grille
abstraite. On déplace et redimensionne les fenêtres, on pose les composants d'un modèle, on
verrouille ce qui ne doit plus bouger.

L'aimant aligne sur les bords et les centres des voisins, avec des guides visibles. Les modèles
de fenêtre sont **à format libre** : un modèle dessiné pour du 16:9 s'applique à une case 4:3
sans que ses éléments se déforment — l'aimant lui-même ne modifie jamais les proportions d'une
fenêtre, ce qu'il a longtemps fait sans que ça se voie.

Tout ce qui est réglable ici l'est aussi par **macro ou déclencheur** : arbre de paramètres
continus pour ce qui se dose, actions discrètes pour ce qui se déclenche. Une capacité pilotable
seulement à la souris est une capacité morte le jour d'une émission.

---

## Ce qui mérite d'être su

**Le coût d'un mur est une affaire de mémoire, pas de calcul.** Il n'est pas limité par la
puissance mais par les déplacements d'octets. Les décisions qui ont payé concernent la taille des
tuiles et la réutilisation des habillages — pas le nombre de fils.

**L'habillage est mis en cache par signature.** Textes, cadres et lampes ne sont re-fabriqués que
lorsque leur rendu change RÉELLEMENT. Sans cela, un tally poussé dix fois par seconde re-fabrique
l'habillage plein cadre dix fois par seconde, et le mur perd une trame à chaque fois.

**Le mode tranche** (`slice_mode`, désactivé par défaut) lit les entrées par tranches et publie
la sortie en commit progressif au lieu d'attendre l'image entière. Un étage qui travaille en
image entière ajoute une trame de latence à toute la chaîne qui le traverse — et cette dette
n'apparaît sur aucun compteur, puisque l'étage affiche une cadence parfaite.

**Le chemin GPU** (CUDA, et `gpu_slice` en option) ne change que le LIEU des octets, jamais le
protocole : mêmes attentes de tranche, mêmes budgets par tuile, même commit progressif.
`force_cpu` le désactive sans que le conteneur détienne la carte — ce qui n'allait pas de soi :
la sonde d'auto-détection ouvrait le périphérique avant même que le réglage soit lu.

---

## Ce qu'il publie

Le conteneur expose ses métriques sur `:8080` — cadence, ventilation du temps de composition par
étage, trames tronquées, latence propre, état GPU, re-fabrications d'habillage par seconde.
Elles ne sont pas décoratives : un choix de placement montre son coût au lieu de le cacher.

Le contrôle live passe par les points d'entrée déclarés dans `plugin.json` : fenêtres, styles,
overlays, horloges, tally, textes.

---

## L'installer

**Depuis Bobi.Studio** — page **Catalogue**, qui liste les composants publiés et les installe.
Ou Réglages → Plugins → *Importer*, avec un paquet `.mxlplugin`.

**À la main** — clonez ce dépôt dans `plugins/multiview/` d'une instance, puis rechargez le
registre des plugins.

> Modifier `hooks.py` exige un rechargement du registre : l'orchestrateur l'importe une fois, au
> scan. Un hook qui ne se déclenche jamais est une panne parfaitement silencieuse.

---

## Le lire

- `script.py` — le plugin, un gabarit `str.format` rendu par l'orchestrateur et exécuté dans le
  conteneur. **Toute accolade littérale y est doublée `{{ }}`**, commentaires compris.
- `hooks.py` — les hooks de cycle de vie, exécutés dans l'orchestrateur.
- `multiview.js` — l'éditeur de mur. `monitoring.html` — la page de supervision.
- `plugin.json` — câblage, schéma de configuration, surface de macros, points de contrôle.
- `meta.json` — le journal des versions, et c'est là que vivent les *pourquoi* : chaque entrée
  dit ce qui cassait, ce qui a été mesuré, et ce que la correction a coûté.

---

## Licence

GPL-3.0-or-later — voir [LICENSE](LICENSE). Copyright © 2026 BOBI SAS, France.
