/* Local-development fallback for window.__APP_CONFIG__.

   In production docker/entrypoint.sh renders this file's contents from the
   container environment into /gen/app-config.js and Caddy serves THAT for
   /app-config.js, so this copy is never reached. It exists so that opening src/
   from a plain file server still works, and so the key names have one readable
   definition to keep the entrypoint in sync with.

   Every value is empty on purpose: empty means "feature off" everywhere it is
   read, so the local-dev default tracks nothing and shows no banner. */
window.__APP_CONFIG__ = window.__APP_CONFIG__ || {
  url: "",            // umami tracker script URL; empty disables analytics
  websiteId: "",
  sri: "",
  dnt: "",            // empty is treated as "honour Do-Not-Track"
  dev: "",            // non-empty shows the DEV banner
  startedAt: "",
  sourceUrl: "",
};
