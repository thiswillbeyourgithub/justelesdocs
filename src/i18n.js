/* Interface strings, in every language the site offers.

   All languages live in one page rather than at separate URLs. A corpus is
   often multilingual and the embedding model is too, so a reader searching in
   French legitimately gets English hits and vice versa: splitting the site by
   language would imply a split in the content that does not exist.

   Two layers. STRINGS below is the SOFTWARE's table, French and English, complete
   and neutral: it names no corpus, so a site with no overlay is fully usable.
   On top of it the CORPUS's overlay (its strings/<lang>.json, compiled by
   scripts/stage.py into site-config.js as window.__SITE__.strings) replaces any
   key it carries and adds its own: the tagline, worked examples a reader of THAT
   corpus would ask, the labels of its facet values and tiers, its footer links.
   The overlay also decides the language list (__SITE__.languages) and the
   default; a language the software has no table for falls back key by key to the
   default language, then to English, so an overlay can add a third language one
   string at a time.

   The choice is remembered per browser and mirrored into <html lang>, which
   matters beyond cosmetics: it drives hyphenation, spell check, and how a screen
   reader pronounces the page.

   Keys are flat and named after what they say, not where they appear, so moving
   a string between views does not rename it. */

import { pdfHref } from "./paths.js";
import { load, save } from "./store.js";

/* What the corpus declares, read once. A missing site-config.js (an unstaged src/
   tree) means the software's own two languages and no overlay. `globalThis` rather
   than `window` so the tests can set it under Node before importing this file. */
const SITE = globalThis.__SITE__ || {};
const LANGS = Array.isArray(SITE.languages) && SITE.languages.length ? SITE.languages : ["fr", "en"];
const DEFAULT_LANG = LANGS.includes(SITE.default_language) ? SITE.default_language : LANGS[0];
const OVERLAY = SITE.strings || {};

const STRINGS = {
  fr: {
    tagline: "Recherche sémantique dans un corpus de documents",
    dev_banner: "Prototype en développement.",
    dev_restarted: "Dernier redémarrage il y a {ago}.",
    dev_deploying: "Déploiement en cours.",
    dev_source: "Code source",
    dev_dismiss: "(cliquer pour masquer)",
    // Elapsed-time units for the dev banner. {s} is the plural mark, which is "s" in
    // both languages for every unit here, so the caller passes it rather than each
    // language carrying two strings.
    ago_now: "moins d'une minute",
    ago_minutes: "{n} minute{s}",
    ago_hours: "{n} heure{s}",
    ago_days: "{n} jour{s}",
    search_label: "Rechercher dans le corpus",
    // A worked example rather than a description of one. The search compares the
    // MEANING of sentences, so two keywords retrieve worse than the question a
    // reader would actually ask, and nothing in the interface says so unless the
    // placeholder does.
    search_placeholder: "Posez une question en une phrase",
    search_button: "Rechercher",
    // The one disclosure above the results. Two texts for the same summary: closed,
    // it is the only thing on the page that can say a filter is on.
    // Le libellé du « ? » posé à côté de la boîte de recherche: le bouton ne porte
    // qu'un caractère, donc c'est cette phrase que lisent le lecteur au survol et le
    // lecteur d'écran.
    syntax_help: "Aide : comment poser la question",
    advanced_summary: "Recherche avancée",
    advanced_narrowed: "Recherche avancée (filtres actifs)",
    // The optional query algebra. The examples are translated rather than shared,
    // because a French reader typing an English example gets English documents.
    syntax_sentence: "Écrivez une vraie phrase plutôt que des mots-clés : la recherche compare le sens des phrases, pas les mots.",
    // Same lesson as the line above, stated where it costs a reader the most: a
    // sigle is short for a human who already knows the field and nearly empty for a
    // model that also has to work in another language and another specialty.
    // Measured, not claimed: at the shipped ranking, a question asked in the
    // document's own language finds the right page in the top ten 70 to 85% of
    // the time, and a question asked in the other language 39% (117 queries,
    // 23 of them crosslingual; scripts/evaluate_rescore.mjs prints the split).
    // A reader who does not know that reads an empty result as "the corpus has
    // nothing on this", which is the one wrong conclusion this page can cause.
    syntax_language: "La recherche traverse les langues : une question dans une langue peut ramener un passage dans une autre. Le modèle reste bien meilleur dans la langue du document, alors posez la question dans la langue des documents que vous visez.",
    syntax_context: "Écrivez les sigles en toutes lettres : un sigle est ambigu pour le modèle, son développement ne l'est pas. Il travaille sur toutes les langues et tous les sujets à la fois, alors chaque mot de contexte en plus aide.",
    syntax_example_minus: "effets du café -(chez l'enfant)",
    syntax_minus: "« -(…) » retire un sujet : cherche les effets du café, en écartant ce qui parle de l'enfant.",
    syntax_example_or: "(effets du café | effets de la caféine)",
    syntax_or: "« | » fond plusieurs formulations en une seule question, des deux côtés du « -(…) ». « (a | b) » et « a +(b) » disent la même chose.",
    query_only_negative: "Il manque ce qu'il faut chercher : « -(…) » retire un sujet d'une question, il ne peut pas en tenir lieu.",
    query_terms: "Trop de termes dans cette question ({n} sur {max} au maximum). Regroupez-en quelques-uns.",
    query_long: "Question trop longue ({n} caractères, {max} au maximum). Posez-la en une phrase.",
    query_short: "Question trop courte ({min} caractères au minimum). La recherche compare le sens de phrases entières : un mot isolé ne lui dit presque rien, une question en dit beaucoup.",
    query_cancelled: "Ce qui est soustrait annule toute la question : il ne reste rien à chercher.",
    loading_index: "Chargement de l'index…",
    index_ready: "{chunks} passages dans {docs} documents",
    searching: "Recherche…",
    no_results: "Aucun passage ne correspond. Essayez des termes plus généraux, ou retirez des filtres.",
    results_count: "{n} passages, {docs} documents",
    // Quand un reclasseur tourne, l'ordre bouge sous les yeux du lecteur pendant quelques
    // secondes: sans cette ligne, une liste qui se réorganise se lit comme un bug.
    // The two ways of stacking one answer, and the corpus listing that stands in for
    // it before any question is asked.
    view_label: "Affichage",
    view_passages: "Par passage",
    view_documents: "Par document",
    // Le score affiché est celui qui a trié la liste, donc décocher change les deux.
    rescore_label: "Tenir compte des mots exacts",
    rescore_title: "Par défaut le classement ne tient compte que de la proximité de sens. Cochez pour ajouter au score une part de correspondance mot à mot (BM25), sur les 50 meilleurs passages.",
    norefs_label: "Écarter les bibliographies",
    norefs_title: "Coché par défaut. Les pages qui ne sont qu'une liste de références (auteurs, revues, années) sont reculées dans le classement, sans jamais être supprimées : elles citent les mots de la question sans y répondre. Décochez pour chercher une référence précise.",
    recent_label: "Privilégier les documents récents",
    recent_title: "Coché par défaut. À pertinence comparable, un document récent passe devant un document ancien, parce qu'un document est souvent remplacé par son édition suivante. L'effet est volontairement léger : un texte plus ancien qui répond mieux reste en tête. Décochez pour un classement qui ignore l'année.",
    figures_label: "Figures :",
    figures_include: "incluses",
    figures_exclude: "exclues",
    figures_only: "seulement",
    figures_title: "Les figures (schémas, graphiques, arbres de décision) sont cherchables grâce à une description rédigée par un modèle d'IA. Par défaut elles concourent avec le texte. « Exclues » ne garde que le texte des documents, « seulement » ne montre que des figures.",
    matching_passages: "{n} passage{s} ici",
    // Le rang d'un passage dans la liste non groupée, et son propre score. Groupés,
    // les passages d'un document perdent leur place dans le classement général : ce
    // libellé la rend, et dit du deuxième passage ce que le score du document ne dit
    // que du premier. « n° » et non « # », qui ne se lit pas en français.
    passage_rank: "n° {n} ({score})",
    browse_count: "Les {n} documents du corpus, par ordre alphabétique.",
    // The two pages, named from the reader's side. The search page is the site's
    // front door; the document list is the other way in, for a reader who knows
    // which document they want and has no question to ask about it.
    nav_browse: "Documents",
    nav_search: "Recherche",
    nav_label: "Pages du site",
    browse_title: "Les documents du corpus",
    filter_name: "Filtrer par nom de document",
    browse_none: "Aucun document ne correspond.",
    browse_intro: "Tout ce que le site sert, par ordre alphabétique. Filtrez la liste, ouvrez un document ou téléchargez-le.",
    landing_browse: "Vous cherchez un document précis plutôt qu'un passage ?",
    landing_browse_link: "Parcourez la liste des documents.",
    page_count: "{n} page{s}",
    filters: "Filtres",
    // The popup that holds a facet's values: its search box and what it says when
    // the search matches nothing. Both live in the popup, never on the page.
    facet_search: "Filtrer la liste",
    facet_none: "Aucune valeur ne correspond.",
    reset_filters: "Réinitialiser",
    page: "page {n}",
    pages: "pages {a} à {b}",
    // The viewer's one document control, and the two answers behind it. The summary
    // names the thing ("PDF complet"), the items name the actions, so the reader
    // reads a noun and then chooses a verb rather than choosing between two verbs
    // before knowing they describe the same file.
    full_pdf: "PDF complet ({n} page{s})",
    full_pdf_unknown: "PDF complet",
    open_in_browser: "Ouvrir dans le navigateur",
    download_file: "Télécharger le fichier",
    // Shown INSTEAD of the two verbs above on a document the site may show but not
    // hand over. The reason is named: a control that is grey and silent reads as a
    // bug, and "pour des raisons de droits" is the actual answer to "why can I not
    // have this file".
    restricted_pdf: "PDF non téléchargeable",
    restricted_reason: "Ce document est sous droits : le site peut en montrer une page, pas en donner le fichier.",
    restricted_pages: "Consultation limitée à la page du passage et à une page de part et d'autre, pour des raisons de droits.",
    // The link to the publisher's own page, and the warning that goes with it. The
    // caveat is in the LABEL and not only in the tooltip, because a tooltip does not
    // exist on a touch screen and this one is not a detail: the URL was found by a
    // language model, and a fabricated but plausible link is exactly what that can
    // produce. data/SOURCE_URL_LOG.tsv records what each one rests on.
    source_link: "Source (IA)",
    source_warning: "Lien vers la page de l'éditeur trouvé par un modèle de langage, puis vérifié automatiquement. Il peut malgré tout être inventé ou renvoyer vers une autre édition : vérifiez-le avant de vous y fier.",
    // The identifier the document prints about ITSELF, unlike source_link above: it
    // was read out of the PDF rather than looked up, so it carries no caveat and no
    // "(IA)". The label is the identifier's own name, because that is what a reader
    // is after when they want to cite the document or find it in a catalogue.
    doi_link: "DOI",
    doi_title: "Ouvrir ce document chez son éditeur via son DOI",
    isbn_link: "ISBN",
    isbn_title: "Chercher ce livre dans les catalogues de bibliothèques (WorldCat)",
    also_in_doc: "{n} autres passages dans ce document",
    other_renditions: "Autre version de ce texte : {list}",
    score: "score",
    // The viewer's own search box: one question, and what it should be asked of.
    search_here: "Rechercher",
    search_scope: "Chercher dans",
    scope_doc: "ce document",
    scope_issuer: "les documents de {issuer}",
    scope_all: "tout le corpus",
    // And, on the search page, what a scoped search says about itself.
    scope_limited: "Recherche limitée à {what}.",
    scope_clear: "Chercher dans tout le corpus",
    passage_expand: "Afficher le passage entier",
    passage_collapse: "Réduire le passage",
    figure_note: "Figure décrite par un modèle d'IA ({model})",
    figure_note_title: "Ce texte n'est pas celui du document : c'est la description d'une figure, rédigée par un modèle d'IA pour la rendre cherchable. Ouvrez la page pour voir la figure elle-même.",
    score_value: "{n} %",
    embed_unavailable: "Le service de recherche est indisponible. Le site reste consultable, mais la recherche sémantique ne fonctionne pas pour le moment.",
    search_busy: "Trop de recherches en même temps : réessayez dans un instant.",
    index_failed: "L'index n'a pas pu être chargé : {error}",
    dim_mismatch: "Incohérence de configuration : le service renvoie des vecteurs de {got} dimensions, l'index en attend {want}. La recherche est désactivée pour éviter des résultats absurdes.",
    theme_toggle: "Thème",
    viewer_loading: "Chargement du document…",
    viewer_failed: "Impossible d'afficher ce document : {error}",
    prev_page: "Page précédente",
    next_page: "Page suivante",
    back_to_results: "Retour aux résultats",
    // The button says what clicking it will do, like the language button, rather than
    // naming its current state.
    // Deux longueurs différentes, volontairement: le bouton est posé sur la page, et
    // une fois le surlignage masqué c'est le mot le plus court qui gêne le moins la
    // lecture de ce qu'il recouvrait.
    hide_highlights: "Cacher le passage",
    show_highlights: "Montrer",
    // Le texte du passage sous la page, replié. Le libellé dit « texte » et non
    // « passage » parce que ce qui s'ouvre est la suite de caractères indexée, pas
    // une citation mise en forme.
    chunk_text: "Texte du passage",
    tour_start: "Visite guidée",
    tour_next: "Suivant",
    tour_done: "Terminer",
    tour_skip: "Passer",
    tour_prev: "Précédent",
    /* The guided tour's captions (tour.js holds the steps, keyed to element ids,
       and reads each caption here). tour_example is what step one types into
       the box and what the search step runs, so it has to be a question the
       corpus can answer: a corpus overlay (strings/<lang>.json) replaces it
       along with the captions that name its own sources. */
    tour_example: "Quels sont les effets du café sur le sommeil ?",
    tour_ask: "Posez une vraie question, comme à un collègue, plutôt qu'une liste de mots-clés, et écrivez les sigles en toutes lettres.",
    tour_metadata: "La recherche lit aussi le titre, l'organisme et l'année de chaque document. Nommez-les pour viser une source.",
    tour_syntax: "Le « ? » à côté de la boîte explique comment poser la question : écrire une phrase plutôt que des mots-clés, développer les sigles, et la syntaxe pour écarter un sujet ou fondre plusieurs formulations.",
    tour_advanced: "« Recherche avancée » réunit les filtres. Ils se construisent à partir des métadonnées disponibles et apparaissent au fur et à mesure que le corpus est documenté.",
    tour_browse: "Vous cherchez un document précis et non un passage ? La liste complète des documents, avec ses propres filtres, est sur sa propre page.",
    tour_search: "Lançons la question de l'étape 1. Chaque résultat est un passage précis, pas un document entier. Plusieurs versions d'un même document sont regroupées. Un bouton bascule l'affichage par passage ou par document.",
    tour_result: "Cliquez sur un titre pour ouvrir la page du PDF, avec le passage surligné. Le document original reste téléchargeable, sauf ceux qui sont sous droits, consultables à la page trouvée et une page de part et d'autre.",
    // The language's own name, shown on the toggle that switches TO it, and the
    // locale its numbers and dates are written in.
    lang_name: "Français",
    locale: "fr-FR",
    issuer: "Émetteur", country: "Pays", year: "Année", language: "Langue",
    doc_type: "Type", topic: "Thème", access: "Fichier",
    facet_access_open: "Téléchargeable",
    facet_access_restricted: "Consultable sur place",

    /* Le sélecteur de niveau, en tête du panneau de filtres. Les libellés des
       niveaux eux-mêmes (guideline_<niveau>) viennent du corpus, qui seul sait ce
       que chacun garde : ils se lisent en escalier, du plus étroit au plus large,
       et un « + » en tête dit « tout le précédent, plus ». Un niveau sans libellé
       s'affiche sous son propre nom. Le titre dit ce que chaque niveau laisse de
       côté, parce que c'est la seule chose qu'un lecteur ne peut pas deviner en
       regardant une liste de résultats. */
    guideline_level: "Niveau",
    guideline_all: "Tous les documents",
    /* Sous le dernier résultat. « {what} » est le libellé du niveau suivant, privé de
       son « + » : les deux lignes se lisent alors comme une phrase, « aussi les
       synthèses et argumentaires ». Le dernier palier a sa propre phrase parce que
       « aussi les tous les documents » ne se lit pas. */
    widen_add: "Élargir la recherche : aussi les {what}",
    widen_all: "Élargir la recherche : tous les documents, quel que soit leur type",
    widen_title: "Le niveau choisi (dans « Recherche avancée ») écarte une partie du corpus avant même le classement. Ce bouton l'élargit d'un cran, et le nouveau niveau est retenu pour les recherches suivantes.",
    guideline_title: "Par défaut, la recherche porte sur tous les documents. Un niveau plus étroit écarte une partie du corpus avant le classement, et chaque niveau garde tout ce que garde le précédent. Un document dont le niveau n'est pas encore renseigné n'est jamais écarté.",
    // The year facet is a two-handle range, so it needs three strings a dropdown
    // does not: one per slider for screen readers, and the readout between them.
    range_from: "Année la plus ancienne",
    range_to: "Année la plus récente",
    range_span: "{a} à {b}",

    /* Facet VALUE labels, keyed facet_<field>_<slug>.

       The manifest stores stable ASCII slugs, because they are a data format: they
       must survive a language switch, sort predictably, and stay put when the
       wording changes. Everything the reader reads is here instead, and a slug
       with no entry falls back to itself (see facetLabel in app.js), so curating a
       new value never breaks the panel.

       The labels of a corpus's own vocabularies (issuers, document types, topics)
       are in its overlay, not here. Issuers there should carry their expansion:
       an acronym identifies a body to someone who already knows it and to nobody
       else, and the whole point of the facet is to let a reader who does not know
       the landscape narrow by who published. */

    // Country and language are ISO codes in the manifest for the same reason, and
    // "FR" in a dropdown is a filing code, not a choice a reader makes.
    facet_country_FR: "France",
    facet_country_UK: "Royaume-Uni",
    facet_country_US: "États-Unis",
    facet_country_CA: "Canada",
    facet_country_AU: "Australie",
    facet_country_INT: "International",
    facet_language_fr: "Français",
    facet_language_en: "Anglais",
    facet_language_ar: "Arabe",


    // Footer, in the same three beats as justelesRCP's: where the content comes from,
    // what this site is not, and who built it. {author}, {agent} and {repo} are filled
    // with links by site.js, which is why the sentence carries them rather than the
    // markup: French and English do not put the credit in the same order.
    foot_disclaimer: "Réutilisation sans caractère officiel, non affiliée aux auteurs des documents. Vérifiez toujours la date d'un document : il peut avoir été remplacé.",
    foot_author: "Site créé par {author}.",
    foot_credits: "Développé avec l'aide de {agent} ; le code de {repo} est sur GitHub.",
    // Both language editions of the author's site exist; the link should land on the
    // one the reader is already reading in.
    foot_author_url: "",
    // The corpus grows by someone pointing out a document that should be in it, so
    // the footer has to ask. {issue} and {contact} are filled with links by site.js,
    // like the credit sentence above.
    foot_suggest: "Un document utile manque, ou une fonctionnalité ? Je suis preneur : ouvrez {issue} ou écrivez-moi via {contact}.",
    foot_suggest_issue: "une issue GitHub",
    foot_suggest_contact: "ma page de contact",
    foot_contact_url: "",
    // The release-notes popup (changelog.js). The bullets themselves are not here:
    // they are written per release in docs/changelog/ and compiled into
    // changelog.json in both languages, because a release note is content, not
    // interface, and it has to be written when the change is made rather than
    // collected later.
    foot_changelog: "Quoi de neuf ?",
    changelog_title_new: "Quoi de neuf ?",
    changelog_title_all: "Journal des versions",
    changelog_since: "Nouveautés depuis votre dernière visite (version {v}).",
    changelog_version: "Version {v}",
    changelog_show_all: "Tout afficher",
    changelog_close: "Fermer",
    changelog_commit: "Voir ce changement dans le code, sur GitHub",
  },
  en: {
    tagline: "Semantic search across a corpus of documents",
    dev_banner: "Prototype under development.",
    dev_restarted: "Last restarted {ago} ago.",
    dev_deploying: "Deployment in progress.",
    dev_source: "Source code",
    dev_dismiss: "(click to dismiss)",
    ago_now: "less than a minute",
    ago_minutes: "{n} minute{s}",
    ago_hours: "{n} hour{s}",
    ago_days: "{n} day{s}",
    search_label: "Search the corpus",
    search_placeholder: "Ask a question in one sentence",
    search_button: "Search",
    syntax_help: "Help: how to ask",
    advanced_summary: "Advanced search",
    advanced_narrowed: "Advanced search (filters on)",
    syntax_sentence: "Write a real sentence rather than keywords: the search compares the meaning of sentences, not the words.",
    syntax_language: "Search crosses languages: a question in one language can bring back a passage in another. The model is still much better in the document's own language, so ask in the language of the documents you are after.",
    syntax_context: "Spell acronyms out: an acronym is ambiguous to the model, its expansion is not. It works across every language and every subject at once, so each extra word of context helps.",
    syntax_example_minus: "effects of coffee -(in children)",
    syntax_minus: "\"-(…)\" takes a subject out: searches for the effects of coffee, setting aside what is about children.",
    syntax_example_or: "(effects of coffee | effects of caffeine)",
    syntax_or: "\"|\" blends several wordings into one question, on either side of the \"-(…)\". \"(a | b)\" and \"a +(b)\" say the same thing.",
    query_only_negative: "Nothing left to search for: \"-(…)\" takes a subject out of a question, it cannot be the question.",
    query_terms: "Too many terms in this question ({n}, and {max} is the most). Try grouping some of them.",
    query_long: "This question is too long ({n} characters, and {max} is the most). Ask it in one sentence.",
    query_short: "This question is too short ({min} characters is the least). The search compares the meaning of whole sentences: a single word tells it almost nothing, a question tells it a lot.",
    query_cancelled: "What is subtracted cancels the whole question: nothing is left to search for.",
    loading_index: "Loading the index…",
    index_ready: "{chunks} passages across {docs} documents",
    searching: "Searching…",
    no_results: "No passage matches. Try broader wording, or clear some filters.",
    results_count: "{n} passages, {docs} documents",
    view_label: "View",
    view_passages: "By passage",
    view_documents: "By document",
    rescore_label: "Count the exact words",
    rescore_title: "By default the ranking goes on meaning alone. Tick to add a word-for-word match (BM25) to the score, over the 50 best passages.",
    norefs_label: "Set bibliographies aside",
    norefs_title: "Ticked by default. Pages that are nothing but a list of references (authors, journals, years) are pushed down the ranking, never removed: they carry the words of the question without answering it. Untick to look for a citation itself.",
    recent_label: "Favour recent documents",
    recent_title: "Ticked by default. Where two passages are about as relevant, the more recent document comes first, because a document is often superseded by its next edition. The effect is deliberately small: an older text that answers better still leads. Untick for a ranking that ignores the year.",
    figures_label: "Figures:",
    figures_include: "included",
    figures_exclude: "left out",
    figures_only: "only",
    figures_title: "Figures (diagrams, charts, decision trees) are searchable through a description written by an AI model. By default they compete with the text. \"Left out\" keeps only the documents' own text, \"only\" shows nothing but figures.",
    matching_passages: "{n} passage{s} here",
    // A passage's rank in the ungrouped list, and its own score. Grouped, a document's
    // passages lose their place in the overall ranking: this label gives it back, and
    // says about the second passage what the document's score only says about the first.
    passage_rank: "#{n} ({score})",
    browse_count: "All {n} documents in the corpus, in alphabetical order.",
    nav_browse: "Documents",
    nav_search: "Search",
    nav_label: "Site pages",
    browse_title: "The documents in the corpus",
    filter_name: "Filter by document name",
    browse_none: "No document matches.",
    browse_intro: "Everything this site serves, in alphabetical order. Filter the list, open a document, or download it.",
    landing_browse: "Looking for one document rather than a passage?",
    landing_browse_link: "Browse the list of documents.",
    page_count: "{n} page{s}",
    filters: "Filters",
    facet_search: "Filter the list",
    facet_none: "Nothing matches.",
    reset_filters: "Reset",
    page: "page {n}",
    pages: "pages {a} to {b}",
    full_pdf: "Full PDF ({n} page{s})",
    full_pdf_unknown: "Full PDF",
    open_in_browser: "Open in the browser",
    download_file: "Download the file",
    restricted_pdf: "PDF not downloadable",
    restricted_reason: "This document is under copyright: the site may show you a page, not give you the file.",
    restricted_pages: "For legal reasons, reading is limited to the passage's page and one page either side.",
    source_link: "Source (AI)",
    source_warning: "Link to the publisher's page found by a language model, then checked automatically. It may still be fabricated or point to a different edition: check it before relying on it.",
    doi_link: "DOI",
    doi_title: "Open this document at its publisher through its DOI",
    isbn_link: "ISBN",
    isbn_title: "Look this book up in library catalogues (WorldCat)",
    also_in_doc: "{n} more passages in this document",
    other_renditions: "Also published as: {list}",
    score: "score",
    // The viewer's own search box: one question, and what it should be asked of.
    search_here: "Search",
    search_scope: "Search in",
    scope_doc: "this document",
    scope_issuer: "documents from {issuer}",
    scope_all: "the whole corpus",
    // And, on the search page, what a scoped search says about itself.
    scope_limited: "Search limited to {what}.",
    scope_clear: "Search the whole corpus",
    passage_expand: "Show the whole passage",
    passage_collapse: "Show less",
    figure_note: "Figure described by an AI model ({model})",
    figure_note_title: "This is not the document's text: it describes a figure, written by an AI model to make it searchable. Open the page to see the figure itself.",
    score_value: "{n}%",
    embed_unavailable: "The search service is unavailable. The site still works, but semantic search is down for now.",
    search_busy: "Too many searches at once: try again in a moment.",
    index_failed: "The index could not be loaded: {error}",
    dim_mismatch: "Configuration mismatch: the service returns {got}-dimensional vectors and the index expects {want}. Search is disabled rather than returning nonsense.",
    theme_toggle: "Theme",
    viewer_loading: "Loading the document…",
    viewer_failed: "This document could not be displayed: {error}",
    prev_page: "Previous page",
    next_page: "Next page",
    back_to_results: "Back to results",
    hide_highlights: "Hide the passage",
    show_highlights: "Show",
    chunk_text: "Passage text",
    tour_start: "Guided tour",
    tour_next: "Next",
    tour_done: "Finish",
    tour_skip: "Skip",
    tour_prev: "Previous",
    tour_example: "How does coffee affect sleep?",
    tour_ask: "Ask a real question, as you would a colleague, rather than a list of keywords, and spell acronyms out.",
    tour_metadata: "The search also reads each document's title, issuer and year. Name them to aim at a source.",
    tour_syntax: "The \"?\" beside the box says how to ask: write a sentence rather than keywords, spell out acronyms, and the syntax for ruling a subject out or blending several wordings into one question.",
    tour_advanced: "\"Advanced search\" holds the filters. They are built from whatever metadata exists, and appear as the corpus gets documented.",
    tour_browse: "Looking for one document rather than a passage? The full list, with its own filters, has a page of its own.",
    tour_search: "Let's run the question from step 1. Each result is one passage, not a whole document. Several renditions of the same document are grouped together. A button switches between listing passages and listing documents.",
    tour_result: "Click a title to open the PDF page itself, with the passage highlighted. The original document stays downloadable, except for the ones under copyright, which are readable at the page found and one page either side.",
    lang_name: "English",
    locale: "en-GB",
    issuer: "Issuer", country: "Country", year: "Year", language: "Language",
    doc_type: "Type", topic: "Topic", access: "File",
    facet_access_open: "Downloadable",
    facet_access_restricted: "Read here only",

    /* The level selector, at the head of the filter panel. Four labels read as a
       staircase, narrowest first: each level keeps everything the one above it
       keeps, and the "+" says so without needing a sentence. The title says what
       each level leaves out, which is the one thing a reader cannot infer from
       looking at a list of results. */
    guideline_level: "Level",
    guideline_all: "All documents",
    widen_add: "Widen the search: also the {what}",
    widen_all: "Widen the search: every document, whatever its type",
    widen_title: "The level (under \"Advanced search\") keeps part of the corpus out before anything is ranked. This widens it by one notch, and the new level is remembered for the searches after this one.",
    guideline_title: "By default the search covers every document. A narrower level keeps part of the corpus out before anything is ranked, and each level keeps everything the previous one keeps. A document whose level is not curated yet is never kept out.",
    range_from: "Earliest year",
    range_to: "Latest year",
    range_span: "{a} to {b}",


    facet_country_FR: "France",
    facet_country_UK: "United Kingdom",
    facet_country_US: "United States",
    facet_country_CA: "Canada",
    facet_country_AU: "Australia",
    facet_country_INT: "International",
    facet_language_fr: "French",
    facet_language_en: "English",
    facet_language_ar: "Arabic",


    foot_disclaimer: "Unofficial reuse, not affiliated with the documents' authors. Always check a document's date: it may have been superseded.",
    foot_author: "Built by {author}.",
    foot_credits: "Developed with the help of {agent}; the {repo} code is on GitHub.",
    foot_author_url: "",
    foot_suggest: "A useful document missing, or a feature? I want to hear about it: open {issue} or write to me via {contact}.",
    foot_suggest_issue: "a GitHub issue",
    foot_suggest_contact: "my contact page",
    foot_contact_url: "",
    foot_changelog: "What's new?",
    changelog_title_new: "What's new?",
    changelog_title_all: "Release notes",
    changelog_since: "What changed since your last visit (version {v}).",
    changelog_version: "Version {v}",
    changelog_show_all: "Show everything",
    changelog_close: "Close",
    changelog_commit: "See this change in the code, on GitHub",
  },
};


/**
 * Pick the starting language: a remembered choice, else the browser's, else the
 * corpus's default.
 *
 * The default wins whenever the browser lists it at all, even behind another
 * offered language: a French reader with an English-first browser still reads the
 * site in the language its corpus and readership are mostly in. Only a browser
 * that does not list the default gets its first offered language instead.
 */
function initial() {
  const saved = load("lang");
  if (saved && LANGS.includes(saved)) return saved;
  const nav = (navigator.languages || [navigator.language || DEFAULT_LANG])
    .map((tag) => String(tag).split("-")[0].toLowerCase());
  if (nav.includes(DEFAULT_LANG)) return DEFAULT_LANG;
  return nav.find((code) => LANGS.includes(code)) || DEFAULT_LANG;
}

let current = initial();

/** The active language code. */
export function lang() { return current; }

/**
 * Look up a string, substituting {name} placeholders.
 * Missing keys return the key itself: a visible "results_count" in the UI is a
 * bug report, whereas an empty string silently hides a whole line.
 */
export function t(key, vars) {
  let s = lookup(current, key) ?? key;
  if (vars) for (const [k, v] of Object.entries(vars)) s = s.replaceAll(`{${k}}`, String(v));
  return s;
}

/**
 * One key in one language, through both layers: the corpus overlay, then the
 * software's table, then the same two for the default language, then English.
 * Undefined when nobody has it.
 */
function lookup(language, key) {
  return OVERLAY[language]?.[key] ?? STRINGS[language]?.[key]
    ?? OVERLAY[DEFAULT_LANG]?.[key] ?? STRINGS[DEFAULT_LANG]?.[key] ?? STRINGS.en[key];
}

/** Every language the site offers, in the corpus's order. */
export function languages() { return [...LANGS]; }

/** Switch language, persist it, and re-apply every data-i18n binding. */
export function setLang(next) {
  if (!LANGS.includes(next)) return;
  current = next;
  save("lang", next);
  document.documentElement.lang = next;
  apply();
  document.dispatchEvent(new CustomEvent("langchange", { detail: next }));
}

/** The next language in the corpus's list, for the toggle; cycles with three or more. */
export function otherLang() { return LANGS[(LANGS.indexOf(current) + 1) % LANGS.length]; }

/**
 * The plural mark for a count, for strings carrying {s}.
 *
 * Both languages take a bare "s" on every noun these strings count (minute, heure,
 * jour, page), so one helper serves both and each string stays single. A language
 * whose plurals do not work that way would need its own key, not a branch here.
 */
export function plural(n) { return n === 1 ? "" : "s"; }

/**
 * The label for a document menu: "PDF complet (37 pages)".
 *
 * It names the FILE rather than an action, because the actions are the items inside
 * the menu it summarises. The page count is the point: the control sits next to a hit
 * on one page of a document that may run to five hundred, so it has to say what
 * "complet" is going to cost before the reader opens it.
 *
 * @param doc  a document record from index/meta.json; `pages` may be blank, since
 *             MANIFEST.tsv is still being curated.
 */
function fullPdfLabel(doc) {
  return pagesLabel(doc, "full_pdf");
}

/**
 * Whether the site may show this document without handing over the file.
 *
 * One reading of one manifest cell, exported because the viewer, the result list and
 * the browse page all have to agree: a document downloadable in one of them and not in
 * the others is a leak, not an inconsistency.
 *
 * @param doc  a document record from index/meta.json.
 * @returns {boolean}
 */
export function isRestricted(doc) {
  return Boolean(doc) && doc.access === "restricted";
}

/* The one note on the page, created on first use and moved from control to control.
   One rather than one per control, because two notes open at once would be two
   answers to a question nobody asked twice. */
let noteEl = null;

/**
 * Say, on click or touch, why a control is there and does nothing.
 *
 * `title` answers this for a mouse, and nothing answers it for a finger: a touch
 * screen has no hover, so a greyed arrow that refuses to turn the page reads as a
 * broken site rather than as a rule about copyright. The note is what the tap gets.
 *
 * The message is read at click time rather than bound once, because the reason can
 * turn on and off under the same control (the pager's arrows are dead at the edge of
 * a restricted window and alive one page in) and because it has to follow a language
 * change with no rebinding.
 *
 * It closes on the next pointer or key anywhere, on scroll, and on a resize, which
 * covers every way a reader can say "read it, thanks" without adding a close button
 * to a sentence.
 *
 * @param {HTMLElement} element - The control that explains itself.
 * @param {() => string} reason - The sentence, or "" when there is nothing to say.
 */
export function tapNote(element, reason) {
  element.addEventListener("click", (event) => {
    const message = reason();
    if (!message) return;
    // The control is dead by definition here, so a click on it must not also count as
    // a click on whatever it sits inside (a <details> menu, a result row).
    event.preventDefault();
    event.stopPropagation();
    showNote(element, message);
  });
}

function showNote(element, message) {
  if (!noteEl) {
    noteEl = document.createElement("div");
    noteEl.className = "tap-note";
    // polite, not assertive: it is an explanation, not an alarm.
    noteEl.setAttribute("role", "status");
    document.body.append(noteEl);
  }
  noteEl.textContent = message;
  noteEl.hidden = false;
  // Fixed to the viewport, under the control and right-aligned with it, then pushed
  // back inside the window: the controls that carry one of these sit in a corner as
  // often as not, and a note that runs off the screen explains nothing.
  const box = element.getBoundingClientRect();
  noteEl.style.top = `${box.bottom + 6}px`;
  noteEl.style.left = "0px";
  const width = noteEl.getBoundingClientRect().width;
  const left = Math.min(Math.max(8, box.right - width), Math.max(8, innerWidth - width - 8));
  noteEl.style.left = `${left}px`;
  const close = () => {
    noteEl.hidden = true;
    removeEventListener("pointerdown", close, true);
    removeEventListener("keydown", close, true);
    removeEventListener("scroll", close, true);
    removeEventListener("resize", close, true);
  };
  // Registered in the next frame, since the click that opened this note is still on
  // its way up and would close it again at once.
  setTimeout(() => {
    addEventListener("pointerdown", close, true);
    addEventListener("keydown", close, true);
    addEventListener("scroll", close, true);
    addEventListener("resize", close, true);
  }, 0);
}

/**
 * The document's own PDF, as one control with two answers folded inside it.
 *
 * "Open the full PDF · Download the full PDF (106 pages)" asked the reader to choose
 * between two words before they had decided anything, and took a line and a half to
 * do it. The menu names the FILE and its length, and only opens the two verbs once
 * the reader has said they want the document at all.
 *
 * Built here rather than on each page because the viewer's bar and the two lists show
 * the same control, and the only difference that matters is which page it sits on. A
 * page that shows exactly one of them passes `ids`, which is what makes the viewer's
 * menu addressable by the browser gate; a list must not, since a hundred rows cannot
 * share one id.
 *
 * @param doc  a document record from index/meta.json.
 * @param {{ids?: boolean}} [options]
 * @returns {HTMLDetailsElement}
 */
export function pdfMenu(doc, { ids = false } = {}) {
  // A restricted document is shown but never handed over, so it gets a control that
  // says so rather than a menu whose two entries are both refused. Not a <details>: an
  // empty menu that opens onto nothing is worse than a label that states the rule.
  if (isRestricted(doc)) {
    const note = document.createElement("span");
    note.className = "pdf-menu restricted";
    note.textContent = t("restricted_pdf");
    note.title = t("restricted_reason");
    // The same sentence the pointer gets from `title`, for the finger that has no
    // pointer. Rebuilt with the control, so it needs no unbinding on a repaint.
    tapNote(note, () => t("restricted_reason"));
    if (ids) note.id = "pdf-summary";
    return note;
  }
  const href = pdfHref(doc);
  const menu = document.createElement("details");
  menu.className = "pdf-menu";
  const summary = document.createElement("summary");
  summary.className = "plain";
  summary.textContent = fullPdfLabel(doc);
  const items = document.createElement("div");
  items.className = "pdf-menu-items";
  const open = document.createElement("a");
  open.href = href;
  open.textContent = t("open_in_browser");
  const download = document.createElement("a");
  download.href = href;
  // The attribute, not a verb in a label: this is what makes the browser save the
  // file instead of navigating to it, and what gives the saved copy its real name.
  download.download = doc.file;
  download.textContent = t("download_file");
  if (ids) {
    summary.id = "pdf-summary";
    open.id = "pdf";
    download.id = "dl";
  }
  items.append(open, download);
  menu.append(summary, items);
  return menu;
}

/**
 * Close every bar menu on the page, except one the reader is inside.
 *
 * <details> does not close itself when the reader looks elsewhere or presses Escape,
 * and a list page can have a hundred of them open at once. Shared so the viewer and
 * the lists cannot drift into two different ideas of when a menu is done. All three
 * kinds are listed here rather than given a common class, because the class is also
 * what styles them and they do not look the same inside: a column of links, a small
 * form, and four paragraphs of prose.
 *
 * @param {EventTarget|null} [inside] the click target, when closing after a click.
 */
export function closeMenus(inside = null) {
  for (const menu of document.querySelectorAll(
    "details.pdf-menu, details.doc-search, details.syntax-help")) {
    if (!inside || !menu.contains(inside)) menu.open = false;
  }
}

/**
 * Whether a string is an absolute http(s) URL.
 *
 * The test every URL from outside the code gets before it becomes a link: a manifest
 * cell, a container variable. Anything else (`javascript:`, a relative path, a typo)
 * is treated as absent, so a bad value costs a missing link rather than a live one.
 *
 * @param {unknown} url
 * @returns {boolean}
 */
export function isHttpUrl(url) {
  return /^https?:\/\//i.test(String(url ?? "").trim());
}

/**
 * The note that marks a figure chunk's text as a model's description, not a quotation.
 *
 * One helper because the result list and the viewer both print it, and what it must
 * say (that a model wrote it, and which) is the part that must not drift between them.
 *
 * @param {{model?: string}} figure - The chunk's `figure` field.
 * @returns {string}
 */
export function figureNote(figure) {
  return t("figure_note", { model: figure.model || "?" });
}

/**
 * An <a> to somebody else's page: a new tab, and no window.opener handed to it.
 *
 * @param {string} href
 * @param {string} text
 * @returns {HTMLAnchorElement}
 */
export function externalLink(href, text) {
  const link = document.createElement("a");
  link.href = href;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = text;
  return link;
}

/**
 * The link to where a document was published, or null when there is none.
 *
 * Built here, next to its two strings, because the result list and the viewer both
 * show it and the WARNING is the part that must not drift between them: the URL
 * comes from a language-model lookup (see DESIGN.md, "Finding where each document
 * was published"), so it can be plausible and wrong, and a reader who follows it
 * has to be told that before they trust what they land on. Two copies of that
 * sentence would eventually be one copy of it.
 *
 * The caveat rides in the visible label as well as in the tooltip: `title` shows on
 * hover and a phone has no hover.
 *
 * @param doc  a document record from index/meta.json; `source_url` is blank on the
 *             documents whose publisher has withdrawn them, and blank is a link the
 *             reader must not get.
 */
export function sourceLink(doc) {
  const url = doc && doc.source_url;
  if (!isHttpUrl(url)) return null;
  const link = externalLink(url.trim(), t("source_link"));
  link.title = t("source_warning");
  link.className = "src-ai";
  return link;
}

/**
 * The link to the identifier a document prints about itself, or null when it prints
 * none. A journal article carries a DOI, a book or an agency report an ISBN, and the
 * three documents in the corpus that carry both get the DOI: it resolves to the work
 * itself rather than to a catalogue record of it.
 *
 * Built here next to sourceLink for the same reason that one is: the result list and
 * the viewer both show it, and one definition is how the two cannot drift apart.
 * Unlike sourceLink it needs no warning, because the value was read out of the PDF
 * rather than looked up by a language model (scripts/manifest.py, find_doi).
 *
 * The cells are VALIDATED here rather than trusted, even though manifest.py wrote
 * them: data/MANIFEST.tsv is hand-curated, NO_IDENTIFIER ("-") is a real value in it
 * meaning "looked, found none", and a link built from a typo would open doi.org on
 * nothing. A DOI is a prefix and a suffix separated by a slash; an ISBN is ten or
 * thirteen digits once its hyphens are gone. Anything else is treated as absent.
 *
 * @param doc  a document record from index/meta.json.
 * @returns {HTMLAnchorElement|null}
 */
export function identifierLink(doc) {
  const doi = String((doc && doc.doi) || "").trim();
  const isbn = String((doc && doc.isbn) || "").replace(/[-\s]/g, "").toUpperCase();
  let href = null;
  let key = null;
  if (/^10\.\d{4,9}\/\S+$/.test(doi)) {
    href = `https://doi.org/${encodeURI(doi)}`;
    key = "doi";
  } else if (/^(\d{9}[\dX]|\d{13})$/.test(isbn)) {
    // WorldCat's `bn:` search rather than its /isbn/ path, which answers a 308, and
    // rather than Open Library, which has no record for the French agency ISBNs that
    // are most of this corpus's books. A search page always exists, even when the
    // catalogue holds nothing: an empty result is a truthful answer, a 404 is not.
    href = `https://search.worldcat.org/search?q=bn%3A${isbn}`;
    key = "isbn";
  }
  if (!href) return null;
  const link = externalLink(href, t(`${key}_link`));
  link.title = t(`${key}_title`);
  link.className = "doc-id";
  return link;
}

/**
 * Where a document can be looked up off this site: its own identifier, then where it
 * was published, whichever of the two exist.
 *
 * In that order on every page that shows them: the identifier was read out of this
 * PDF rather than looked up, so it is the more trustworthy of the two and belongs
 * nearer the file it describes, and it is what a reader copies to cite the document.
 * The publisher link leaves the site and says so, so it comes last.
 *
 * @param doc  a document record from index/meta.json.
 * @returns {HTMLAnchorElement[]} zero, one or two links, for the caller to separate.
 */
export function docRefLinks(doc) {
  return [identifierLink(doc), sourceLink(doc)].filter(Boolean);
}

/** Body of `fullPdfLabel`: `<key>` with a page count, or `<key>_unknown`. */
function pagesLabel(doc, key) {
  const pages = Number(doc && doc.pages);
  // MANIFEST.tsv is curated by hand, so a blank page count is a state the UI has to
  // have a sentence for rather than a reason to print "NaN pages".
  if (!Number.isFinite(pages) || pages < 1) return t(`${key}_unknown`);
  return t(key, { n: pages, s: plural(pages) });
}

/**
 * Fill every element carrying data-i18n (text) or data-i18n-attr (attributes).
 * Declarative so static markup stays readable and translatable without JS logic:
 *   <h1 data-i18n="tagline"></h1>
 *   <input data-i18n-attr="placeholder:search_placeholder,aria-label:search_label">
 */
export function apply(root = document) {
  root.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  root.querySelectorAll("[data-i18n-attr]").forEach((el) => {
    for (const pair of el.dataset.i18nAttr.split(",")) {
      const [attr, key] = pair.split(":").map((s) => s.trim());
      if (attr && key) el.setAttribute(attr, t(key));
    }
  });
  // The toggle names the language it switches TO, in that language's own words
  // ("English" on a French page), and disappears on a one-language site.
  const toggle = root.getElementById?.("lang-btn");
  if (toggle) {
    toggle.textContent = lookup(otherLang(), "lang_name") ?? otherLang();
    toggle.hidden = LANGS.length < 2;
  }
  document.documentElement.lang = current;
}
