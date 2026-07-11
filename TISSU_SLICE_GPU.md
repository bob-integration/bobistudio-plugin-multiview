# Tranche GPU du multiview — design + banc gate (chantier TISSU_SLICE, phase GPU)

> Suite de TISSU_SLICE.md §4 (décision : migrer le GPU, APRÈS le tissu CPU). Le slice CPU
> (0.24.x/0.25.x, cadence « flow ») est la RÉFÉRENCE SÉMANTIQUE — le GPU ne change QUE le
> lieu des octets (VRAM), jamais le protocole (attentes get_slice, budgets par tuile,
> backoff, ciblage d'index d'epoch, commit progressif).
> État : squelette 0.26.0 (flag `gpu_slice`, défaut OFF) ; **banc gate PASSÉ (T4 dl360Horace,
> sous charge) : verdict GO-MEGA-135** — bandes fines 36 l REJETÉES (trame 15,26 ms, le coût de
> LANCEMENT des kernels domine : compose 10,6 ms à 30 lancements vs 0,71 ms pleine trame) ;
> méga-bandes 135 l : trame 5,66 ms, 1ʳᵉ bande 0,58 ms, surcoût transferts +0,12 ms, D2H
> recouvert ~0,44 ms gagnés. **Micro-batch (§b var.2) IMPLÉMENTÉ en 0.27.0** : `gpu_batch_bands`
> (défaut 4 bandes/lot = 144 l ≈ 135 mesuré), voir §Micro-batch livré en bas.

## 0. Pipeline GPU actuel (whole-frame, référence à battre)

Par trame (banc Phase 0, T4) : gather des tuiles en RAM (vues shm → numpy) → concat dans un
staging hôte ÉPINGLÉ persistant → **1 seul H2D** (`_place_batch`) → resize+place+blends en
VRAM (chrome pré-calculé résident, VU/horloges uploadés en petites tuiles) → **1 seul D2H**
épinglé (`.get(out=pinned)`, ~1,3 ms vs 5,8 ms non épinglé) → copie vers le grain shm.
Leçon Phase 0 gravée dans le code : **un H2D par tuile, ou pageable, fait RÉGRESSER le GPU**
sous le chemin CPU. Toute la question du slice GPU est de savoir si cette leçon tue le
découpage en 30 bandes — ou si elle ne s'appliquait qu'aux transferts pageables/dispersés.

Ordres de grandeur (1080p50, 4:2:2 8 bits) : trame = 4,15 Mo (Y 2,07 + U/V 2×1,04).
Entrées 4×1080p = 16,6 Mo/trame. PCIe gen3 x16 épinglé ≈ 11-12 Go/s ⇒ H2D groupé entrées
≈ 1,4-1,5 ms, D2H sortie ≈ 0,35 ms. Une bande de 36 lignes de sortie = 138 Ko ; côté
entrées, une bande de sortie tire ~552 Ko (4 tuiles). 30 bandes ⇒ 30 H2D de ~552 Ko
+ 30×3 D2H de ~46 Ko : le coût marginal est dominé par la LATENCE de lancement/transfert
(~5-15 µs/appel cuMemcpyAsync + sync), pas par les octets — c'est exactement ce que le
banc gate doit trancher.

## (a) H2D par bande : LE banc gate (GO/NO-GO, dl360-2/T4)

Script : `tools/bench_gpu_slice_gate.py` (autonome, cupy seul, aucun accès MXL/orchestrateur).

**Quoi mesurer** (médianes sur 300 reps après 50 de warmup, events CUDA pour le pur GPU,
`perf_counter`+sync pour les chemins mixtes hôte) :

1. **M1 — H2D entrées** : 4 tuiles 1080p (16,6 Mo) —
   - groupé pageable (anti-référence Phase 0) ;
   - groupé épinglé 1× (référence actuelle) ;
   - bandé épinglé synchrone : pour `band_lines ∈ {36, 72, 135, 270, 540}` →
     {30, 15, 8, 4, 2} transferts groupés PAR BANDE (toutes tuiles concaténées, comme
     `_gpu_place_band`) ;
   - bandé épinglé sur 2 streams (double-buffer, recouvrement copie/copie) — mesure le
     plafond si on pipeline.
2. **M2 — D2H sortie** : trame 1080p (4,15 Mo) — groupé épinglé 1× (référence) vs bandé
   épinglé synchrone (mêmes tailles de bande, 3 plans par bande) vs bandé 1 stream + commit
   décalé d'une bande (recouvrement D2H/compose simulé).
3. **M3 — compose de bande** : placement (slice-assign device→canvas) + blend_pre d'un
   chrome plein écran + blend de 4 tuiles VU, PAR bande vs pleine trame — mesure le surcoût
   de lancement kernel ×30 (cupy émet ~5-10 kernels par bande ; à 30 bandes × 50 fps =
   jusqu'à 15 k lancements/s — c'est le 2ᵉ risque après les memcpy).
4. **M4 — bout-en-bout simulé** : boucle « gather hôte→épinglé, H2D bande, place, blends,
   D2H bande, écriture mmap simulée (numpy) » séquentielle par bande, pour chaque taille de
   bande — rapporte : temps trame TOTAL et **latence de la 1ʳᵉ bande de sortie** (l'enjeu
   du chantier : le CPU slice sort la 1ʳᵉ bande en ~2 ms).

**Verdict chiffré** (imprimé par le script, budget T=20 ms à 50p) :

| Verdict | Critères (tous requis) |
|---|---|
| **GO fin (36 lignes)** | surcoût M1+M2 bandé(36) vs groupé ≤ **+1,5 ms/trame** ET M4(36) : temps trame ≤ **12 ms** ET 1ʳᵉ bande sortie ≤ **3 ms** |
| **GO méga-bandes** | sinon, ∃ `band_lines ∈ {135, 270, 540}` tel que surcoût M1+M2 ≤ **+0,5 ms** ET M4 : temps trame ≤ 12 ms ET 1ʳᵉ sortie ≤ **8 ms** (garde ~10-15 ms de gain vs whole-frame, cf. TISSU_SLICE §4) |
| **NO-GO** | aucun des deux ⇒ GPU reste whole-frame ; les nœuds tissu GPU passent en `force_cpu` + slice CPU (chemin éprouvé 0.25.x) |

Justification des seuils : le slice CPU tient own ≈ 4-5 ms/trame et 1ʳᵉ bande à ~2 ms ;
le GPU ne se justifie que s'il reste ≤ 12 ms de mur (marge 8 ms pour les attentes get_slice
qui, elles, ne se compressent pas) et s'il conserve l'essentiel du gain de latence — sinon
autant sharder en CPU. Le surcoût transfert est comparé à la référence groupée MESURÉE sur
la même machine (pas aux chiffres théoriques ci-dessus).

**Piège de mesure** : épingler le staging AVANT la boucle (l'alloc pinned coûte ~ms) ;
figer les horloges GPU si possible (`nvidia-smi -lgc`) ; le T4 throttle vite — intercaler
`sleep 0.5` entre sections et rapporter aussi p90.

## (b) Où va la barrière : bande-à-bande retenu, micro-batch en repli

Deux candidats :

1. **Compose par bande dès que la bande d'entrée est complète** (RETENU pour le squelette) :
   structure IDENTIQUE au CPU — la boucle `_compose_bands` attend (get_slice) puis, au lieu
   de poser les rangées dans un canvas numpy, les met en staging et fait 1 H2D + placement +
   blends VRAM. Avantages : un seul chemin de code (budgets/backoff/flow gratuits), latence
   minimale par bande, et le point de greffe streams est propre (H2D bande k+1 pendant
   compose bande k : events cupy, double staging). Inconvénient : 30× le coût fixe de
   lancement — exactement ce que M1/M3 mesurent.
2. **Micro-batch de k bandes** : accumuler k bandes complètes puis 1 H2D/compose/D2H pour
   le lot. C'est STRUCTURELLEMENT la même chose que « méga-bandes » (band_lines = k×36) vu
   du GPU, mais en gardant le get_slice/commit au grain de 36 lignes côté MXL. Si le banc
   vote « GO méga-bandes », l'implémentation préférée est celle-ci : `gpu_batch_bands=k`
   dans `_compose_bands` (staging accumulé sur k bandes, commit `validSlices=(j+1)·k`… avec
   le dernier lot partiel) — l'amont/aval MXL ne voit AUCUNE différence de protocole,
   seule la granularité de commit de CE nœud grossit.

Décision : pas de 3ᵉ voie « kernel persistant / graphe CUDA » à ce stade (complexité sans
mesure). Les clamps/budgets/backoff restent INCHANGÉS dans les deux variantes — ils vivent
dans la partie attente, avant tout octet GPU.

## (c) D2H progressif (commit au fil de l'eau)

Squelette : D2H **synchrone par bande** (`canvas[b0:b1].get(out=pinned)` puis copie vers la
vue grain, puis `commit(validSlices=k+1)`). C'est correct par construction : le commit ne
part qu'avec les octets posés.

Optimisation à greffer SI le banc montre que le D2H sync coûte (M2) : **recouvrement à
profondeur 1** — lancer le D2H de la bande k sur un stream dédié (`get` async vers le
pinned k%2), composer la bande k+1, puis `event.synchronize()` + copie pinned→grain + commit
de k. Coût : +1 bande de latence de commit (0,72 ms à 36 lignes) contre le masquage complet
du D2H. Le point de greffe est marqué dans `_compose_bands` (commentaire « ancre (c) »).
NB : le `.get` DIRECT dans le mmap du grain est proscrit (non épinglé, ~4× plus lent, banc
Phase 0) — toujours transiter par le pinned.

## (d) Interaction avec cadence=flow — rien à changer, par construction

Le flow (TISSU_SLICE §6bis) vit ENTIÈREMENT en amont des octets : tick sur la grille TAI,
ciblage de l'index d'epoch `fi_out` (open_grain à cet index), suivi du grain fi_out par
tuile, budgets par tuile, backoff. Le squelette GPU réutilise `_compose_bands` TEL QUEL —
les seules différences sont DANS la bande (staging/H2D/VRAM/D2H). Conséquences à surveiller
au banc réel (pas au banc gate) :

- le budget par tuile (T ns) et le garde-fou global (1,5×T) datent du CPU : si le coût GPU
  par bande (hors attentes) dérive, les attentes disponibles diminuent — le compteur
  `waits/fallbacks` (:8080 `slice`) le montrera ; ne PAS retoucher les budgets avant mesure ;
- le rattrapage de grille « immédiat si retard < 1 période » (piège n°1 du banc B1) reste
  valable — le GPU ne change pas la phase d'arrivée des grains ;
- `own_latency_ms` exclut déjà les attentes get_slice (P0) ; les temps GPU (H2D/compose/D2H)
  sont du TRAVAIL et restent dedans — c'est ce que le tissu doit lire pour sharder.

## Squelette livré (0.26.0, flag `gpu_slice`, défaut OFF)

- Éligibilité : `SLICE_ON` accepte GPU si `gpu_slice` (sinon inchangé : GPU ⇒ whole-frame) ;
  `GPU_SLICE = GPU and SLICE_ON`. Portrait/hauteur non divisible : mêmes exclusions que CPU.
- Ancre (a) : `_gpu_place_band` — staging hôte épinglé persistant (`_slgpu`, croissance ×2),
  1 H2D groupé PAR BANDE, placement VRAM par slice-assign.
- Ancre (b) : blends chrome (opérandes déjà résidents VRAM via `_to_xp` au bake) + VU/horloges
  (uploadés 1×/trame en tête de `_compose_bands`) par bande, en VRAM — code de blend commun
  CPU/GPU (backend-agnostique).
- Ancre (c) : D2H par bande `.get(out=pinned)` + copie grain + commit progressif, synchrone.
- Métrique `:8080` : `gpu_slice: true/false`.
- Entrées entrelacées / whole-frame : passent par `_place_batch` (upload groupé) comme avant,
  puis `_compose_bands` les traite en dégénéré (totalSlices=1) — cas mixte couvert.

## Lancer le banc demain (dl360-2/T4)

```bash
# dans le conteneur GPU (image bobi-compute-gpu, cupy présent) :
docker run --rm --gpus all -v /opt/bobistudio/plugins/multiview/tools:/b \
  bobi-compute-gpu python3 /b/bench_gpu_slice_gate.py --json /b/gate_result.json
# variantes : --tiles 4 --width 1920 --height 1080 --fps 50 --reps 300
```
Le script imprime les tableaux M1-M4 et le VERDICT (GO fin / GO méga-bandes N lignes /
NO-GO) selon les seuils ci-dessus. Ensuite, selon verdict : déployer un multiview de banc
avec `slice_mode=true, cadence="flow", gpu_slice=true` (+ `slice_lines=<méga>` le cas
échéant) et comparer `own_latency_ms`, `slice.valid0/waits/fallbacks` et la phase de
1ʳᵉ bande (mxl_bench) au même mur en `force_cpu`.

## Micro-batch livré (0.27.0, post-verdict GO-MEGA-135)

- **`gpu_batch_bands`** = nombre de bandes MXL (slice_lines) par LOT GPU, défaut **4** (dernier
  lot partiel si nb % k ≠ 0). Sémantique retenue (vs « nb de lots/trame ») : stable quelle que
  soit la hauteur de sortie, collée au grain MXL — à 1080p/36 l, k=4 → 8 lots de 144 l
  (7×4 + 1×2) ≈ les 135 l mesurés GO (1080/8 n'est pas un multiple de 36). k=1 = squelette.
- **Protocole INCHANGÉ** : attentes get_slice par bande de 36 l (budgets/backoff/fi_out avant
  tout octet GPU) ; commits MXL progressifs au grain 36 l. Seul le GPU grossit au lot : 1 H2D
  groupé épinglé/lot, kernels/lot, **D2H recouvert** (stream dédié non-bloquant + double-buffer
  épinglé : D2H du lot j pendant le compose du lot j+1 ; +1 lot de phase de commit).
- 3 optimisations imposées par le banc micro-batch (le squelette naïf plafonnait à 25 ms) :
  placement **coalescé au lot** (1 gather hôte + 1 slice-assign par tuile/plan/LOT — par bande,
  ~7 ms/trame de lancements) ; fast-path ratio entier = upload de rangées décimées **pleine
  largeur** (copies hôte contiguës ; le gather colonne-stridé coûtait ~5,6 ms/trame) avec
  décimation colonne EN VRAM (csx/csc, octet-identique) ; blends **fusionnés**
  (ElementwiseKernel) **in-place** dans la vue canvas (profite aussi au whole-frame :
  12,4 → 7,9 ms).
- **Banc 3 modes** `tools/bench_gpu_batch.py` (4×1080p → mur 2×2, chrome + 4 VU, 300 trames,
  T4 partagée ~68 %) : cpu_slice 19,4/20,3 ms (p50/p90), 1ʳᵉ bande 0,87 ms ; gpu_whole
  7,9/8,2 ms ; **gpu_batch k=4 : 10,7/11,8 ms, 1ʳᵉ bande 2,38/2,42 ms** — critères GO tenus
  sous charge. Sweep : k=5-8 → trame 8,5-7,5 ms mais 1ʳᵉ bande 2,4→3,6 ms (réglable).
- **Équivalence octet vérifiée** (tolérance 0) : cpu_slice == gpu_whole == gpu_batch.

## Points ouverts

1. ~~Verdict du banc gate~~ FAIT : GO-MEGA-135 (ci-dessus).
2. ~~Si GO : D2H recouvert~~ FAIT (0.27.0, stream dédié, profondeur 1). Reste éventuellement le
   H2D pipeliné (upload lot j+1 pendant compose lot j) — non mesuré nécessaire à ce stade.
3. ~~Si GO méga-bandes : `gpu_batch_bands`~~ FAIT (0.27.0, défaut 4).
4. Budgets par tuile sous coût GPU par lot : à re-regarder sur banc réel (compteurs slice).
5. Rotation portrait sous slice (exclue CPU comme GPU) : hors périmètre, statu quo.
6. Banc réel en conteneur managé (slice_mode+flow+gpu_slice vs force_cpu sur mur live) : à
   faire à la prochaine fenêtre — le banc autonome ne rejoue pas les attentes get_slice.
