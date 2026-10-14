<!-- Version française. English version: README.md
     IMPORTANT : README.fr.md (FR) et README.md (EN) doivent rester synchronisés.
     Quand vous modifiez l'un, mettez l'autre à jour en conséquence. -->

# justelesdocs

*Read this in [English](README.md).*

**Un site statique, sans publicité, qui sert un corpus de PDF choisis avec une recherche sémantique multilingue.** On pose une question en langage naturel, et la réponse s'affiche **sur la page du PDF d'origine, passage surligné**. Aucun modèle de langage ne tourne pour répondre : seulement des plongements (embeddings) et une recherche des plus proches voisins.

justelesdocs est le logiciel. Les documents, leurs métadonnées et leur configuration vivent dans un répertoire séparé, le *corpus*, que le logiciel lit via la variable `CORPUS_DIR`. Une même installation du logiciel peut servir n'importe quel corpus.

## Sommaire

- [Ce que fait le site](#ce-que-fait-le-site)
- [Comment ça marche](#comment-ça-marche)
- [Utilisation avec un assistant IA](#utilisation-avec-un-assistant-ia)
- [Démarrage rapide](#démarrage-rapide)
- [Lancer en local](#lancer-en-local)
- [Déployer avec docker compose](#déployer-avec-docker-compose)
- [Configuration](#configuration)
- [Tests et contrôles](#tests-et-contrôles)
- [Crédits](#crédits)
- [Licence](#licence)

## Ce que fait le site

- On pose une question, pas des mots-clés. La recherche porte sur tous les passages de tous les documents à la fois.
- Un résultat ouvre la page du PDF dont il vient, passage surligné. Plusieurs passages sur une même page ont chacun leur surlignage. De là, un seul bouton ouvre le PDF complet ou le télécharge.
- Les filtres (éditeur, pays, année, langue, type de document, thème, accès) viennent des métadonnées du corpus. Les vocabulaires sont fermés, un champ à plusieurs valeurs correspond dès qu'une valeur correspond, et l'année est un curseur à deux bornes.
- Plusieurs versions d'un même document (une synthèse, le texte complet, un argumentaire) se replient sur la mieux classée, les autres étant proposées à côté.
- Modèle multilingue : une question dans une langue peut trouver un passage dans une autre. L'interface est bilingue français et anglais, et un corpus peut ajouter ou remplacer des textes.
- Un niveau `access: restricted` : certains documents se lisent sans être remis. Seules les pages du passage et deux pages de part et d'autre sont proposées au lecteur, découpées dans le PDF à chaque requête, et le fichier lui-même n'est jamais accessible.
- Les figures sont cherchables grâce à des descriptions écrites hors ligne par un modèle de vision, présentées comme telles.
- Rien n'est traqué. La question part une fois vers le service de recherche du site, qui ne journalise ni la question ni les résultats. La mesure d'audience est désactivée sauf configuration, et même alors elle n'enregistre qu'une vue de page, sans la question. Les pages destinées aux assistants IA comptent une vue de page par requête côté serveur, avec le chemin, le nom de l'assistant et sans la question.

## Comment ça marche

Tout ce qui est lourd se fait hors ligne, sur une machine avec GPU : OCR des pages scannées, découpage de chaque PDF en passages avec leurs coordonnées exactes sur la page, calcul des vecteurs, construction de l'index. Au moment de servir, rien de lourd ne tourne.

La chaîne de construction est une suite d'étapes `uv run scripts/*.py`, chacune protégée par une empreinte de contenu : relancer toute la chaîne sans changement prend moins d'une minute, et un nouveau document ne recalcule que ce qui le concerne. Des contrôles bloquants refusent au lieu d'avertir : aucun PDF sauté, aucun document vide, aucun document restreint sous la racine web, un classement qui retrouve ses propres passages.

Au moment de servir, trois conteneurs durcis tournent : Caddy (fichiers statiques, CSP stricte, limite de débit par IP), un service de recherche Node qui garde l'index en mémoire et classe, et un service de pages Python qui découpe une page d'un document restreint. L'encodeur des questions est externe (le conteneur `embed` du projet frère justelesRCP) ou, en option, dans la même pile (le profil compose `embed`). L'index est binaire, 1024 dimensions à un bit chacune, ce qui tient sur un petit serveur sans GPU.

Pour l'architecture détaillée, voir [ARCHITECTURE.md](ARCHITECTURE.md) ; pour les décisions techniques et leurs mesures, [DESIGN.md](DESIGN.md).

## Utilisation avec un assistant IA

Un site déployé est utilisable par les IA. Collez une fois son lien `/llms.txt` dans Claude, ChatGPT, Le Chat de Mistral, Gemini ou tout assistant capable de lire une page web, et demandez-lui de retenir le site. La page de recherche le dit sous le champ de saisie, avec le lien.

Cette page, écrite par `stage.py` pour le corpus servi, indique à l'assistant ce que contient le corpus et comment se servir de trois accès en texte brut :

- `GET /api/search.md?q=<question>` : la même recherche que la page, avec une réponse en Markdown : les dix meilleurs passages, les métadonnées de leurs documents et des liens. Elle accepte exactement les paramètres de la page de recherche (filtres, intervalle d'années, choix de classement), donc une adresse copiée depuis la barre d'adresse fonctionne une fois son chemin changé. Le débit par IP y est plus limité que pour la recherche de la page, puisque c'est elle qu'un robot viserait.
- `/api/doc/` : la liste des documents, puis une page Markdown par document (métadonnées, liens, premiers mots de chaque page) et une par page (son texte), écrites lors de la mise en place et servies comme fichiers statiques.
- les liens de la visionneuse, pour donner à une personne le passage surligné sur sa page.

Deux boutons font la même chose pour un lecteur : « Copier pour une IA », dans le menu de la visionneuse, copie le passage, sa référence et ces liens, et celui au-dessus des résultats copie l'adresse Markdown de la recherche affichée.

Le niveau `access: restricted` tient sa promesse ici aussi : un document restreint n'a ni page Markdown ni texte, et la recherche ne cite que le seul passage que la page montrerait. `check_served.py` refuse une mise en place où le texte d'un document restreint apparaîtrait sous `api/doc/`. `robots.txt` demande aux robots d'indexation de ne pas visiter `/api/`, tout en y laissant entrer les agents avec lesquels un assistant lit une page quand une personne le lui demande (Claude-User, ChatGPT-User, etc.), et chaque réponse Markdown porte `X-Robots-Tag: noindex`.

**Pourquoi pas de serveur MCP ?** Le [Model Context Protocol](https://fr.wikipedia.org/wiki/Model_Context_Protocol) est l'autre façon courante de brancher un site sur un assistant IA, mais il exige un programme serveur qui tourne à côté du site, un connecteur que chaque utilisateur doit ajouter à la main, et un entretien continu au fil des évolutions du protocole. C'est aussi davantage de surface d'attaque sur un serveur censé rester modeste. justelesdocs suit plutôt la convention [`/llms.txt`](https://llmstxt.org) : des fichiers en texte brut et un accès GET que tout assistant capable de lire une page web peut utiliser sans rien installer.

Réalisé avec Claude Code.

## Démarrage rapide

1. Copiez le corpus d'exemple dans un répertoire à vous, hors de ce dépôt ou dans `local/` (ignoré par git) :

   ```sh
   mkdir -p local/moncorpus && cp corpus.example/corpus.toml local/moncorpus/
   export CORPUS_DIR=local/moncorpus
   ```

   `CORPUS_DIR` est obligatoire et n'a pas de valeur par défaut : chaque chemin `data/...` et `dist/...` est résolu à l'intérieur.

2. Modifiez `$CORPUS_DIR/corpus.toml` : nom du site, filtres, vocabulaires, niveaux, versions, langues. Chaque table est commentée.

3. Déposez vos PDF dans `$CORPUS_DIR/data/GUIDELINES/`. Seul ce répertoire est lu, indexé et servi.

4. Lancez la chaîne :

   ```sh
   uv run scripts/check_pdfs.py
   uv run scripts/manifest.py      # data/MANIFEST.tsv, ne remplit que les cases vides
   uv run scripts/ocr.py           # si certaines pages sont des scans
   uv run scripts/chunk.py
   uv run scripts/verify_chunks.py
   uv run scripts/chunk.py --page-chunks --out data/chunks-page
   uv run scripts/embed.py --chunks data/chunks-page --out-name page-meta-gpu --weights model.onnx --gpu
   uv run scripts/embed.py --out-name meta-gpu --weights model.onnx --gpu
   uv run scripts/build_index.py --dims 1024
   uv run scripts/stage.py
   uv run scripts/check_served.py
   node scripts/check_search.mjs
   ```

   Les deux étapes `embed.py` veulent un GPU : quelques minutes avec, quelques heures sans. Les poids du modèle viennent du projet frère justelesRCP (`scripts/download-model.sh --keep-fp32`).

## Lancer en local

```sh
cd ../justelesRCP && uv run src/embed-service.py --port 8461 --no-backlog   # l'encodeur
EMBED_URL=http://127.0.0.1:8461 INDEX_DIR=$CORPUS_DIR/dist/index node server/service.mjs
PAGES_DIR=$CORPUS_DIR/dist/restricted INDEX_DIR=$CORPUS_DIR/dist/index uv run server/pages.py
uv run scripts/dev_server.py    # le site sur http://127.0.0.1:8649
```

Aucun de ces services ne se recharge quand un fichier change. Après une modification, vérifiez que l'ancien processus ne tient plus son port (`ss -ltnp`).

## Déployer avec docker compose

Le serveur a besoin de `docker/`, `server/` et des modules de `src/` que l'image de recherche copie, plus l'arborescence construite `dist/` (`www/`, `index/`, `restricted/`), par défaut à côté de `docker/` (`DIST_DIR=../dist`).

```sh
cp docker/env.example docker/.env      # puis modifiez-le
sudo docker network create justeles-embed   # une fois, le réseau vers l'encodeur
cd docker && sudo docker compose up -d --build
```

Le site écoute sur `127.0.0.1:8648` et attend un proxy inverse TLS devant lui. `scripts/smoke_deployed.sh <url>`, envoyé au serveur (`ssh ... "sh -s -- <url>" < scripts/smoke_deployed.sh`), vérifie que la recherche répond et qu'un document restreint n'est servi que page par page.

## Configuration

- `$CORPUS_DIR/corpus.toml` : tout ce qui concerne le corpus (`[site]`, `[facets]`, `[vocabularies.*]`, `[tiers]`, `[renditions]`, `[languages.*]`, `[figures]`, `[ocr]`, `[manifest]`, `[judge]`). [corpus.example/corpus.toml](corpus.example/corpus.toml) documente chaque clé.
- `$CORPUS_DIR/strings/<langue>.json` : remplacements facultatifs des textes de l'interface définis dans `src/i18n.js`.
- `$CORPUS_DIR/scenarios.json` : ce que les contrôles attendent de ce corpus (questions d'exemple, documents à trouver) ; sans lui, ils se rabattent sur ce que contient l'index.
- `$CORPUS_DIR/changelog/` et `$CORPUS_DIR/VERSION` : s'ils existent, ils remplacent les notes de version et la version du logiciel sur le site.
- `docker/.env` (à partir de [docker/env.example](docker/env.example)) : `SITE_ID` (nom du projet compose et préfixe des conteneurs, `justelesdocs` par défaut ; deux sites sur un même hôte en veulent deux), `DIST_DIR`, `NETWORK_SUBNET`, `EMBED_NETWORK`, `PORT`, `BIND_ADDR`, mesure d'audience, limites de la recherche.

## Tests et contrôles

```sh
uv run tests/run.py
```

lance les tests du logiciel contre le corpus d'exemple, sans corpus réel ni index, plus `$CORPUS_DIR/tests` quand la variable est définie. Activez le hook de pre-push une fois par clone avec `git config core.hooksPath .githooks` : il lance les tests, et les contrôles du corpus quand `CORPUS_DIR` est défini.

Les contrôles complètent les tests : `verify_chunks.py`, `check_served.py` et `check_search.mjs` dans la chaîne de construction, `check_ui.mjs` et `check_chrome.mjs` dans un navigateur après toute modification de `src/`, `smoke_deployed.sh` contre le site en service.

## Crédits

Développé avec [Claude Code](https://claude.com/claude-code). Rendu des PDF par [pdf.js](https://mozilla.github.io/pdf.js/) (Apache 2.0), embarqué dans `vendor/pdfjs/`.

## Licence

[GNU Affero General Public License v3.0](LICENSE). L'AGPL couvre l'usage à travers un réseau : si vous faites tourner une version modifiée comme site public, ses lecteurs doivent pouvoir obtenir votre code source modifié. Le lien vers le code source en pied de page est la façon dont ce site le propose, donc faites pointer `[site] repo_url` de votre `corpus.toml` vers votre propre dépôt si vous modifiez le code.

La licence ne couvre que le logiciel. Un corpus (ses documents, ses métadonnées et ses textes) n'en fait pas partie et garde les conditions propres à ses documents.
