// Node ESM loader hook that redirects the CDN specifiers used by
// src/js/zarr_file.js to the equivalent, version-pinned packages installed
// locally under tests/js/node_modules, so pyodide-driven tests don't depend
// on jsdelivr being reachable/unchanged at test time.

const CDN_TO_LOCAL = {
  'https://cdn.jsdelivr.net/npm/zarrita/+esm': 'zarrita',
  'https://cdn.jsdelivr.net/npm/@zarrita/storage/zip/+esm': '@zarrita/storage/zip',
  'https://cdn.jsdelivr.net/npm/@zarrita/storage/fetch/+esm': '@zarrita/storage/fetch',
  'https://cdn.jsdelivr.net/npm/fast-xml-parser/+esm': 'fast-xml-parser',
};

export async function resolve(specifier, context, nextResolve) {
  const mapped = CDN_TO_LOCAL[specifier];
  if (mapped) {
    // resolve relative to this loader file, so tests/js/node_modules is used
    return nextResolve(mapped, { ...context, parentURL: import.meta.url });
  }
  return nextResolve(specifier, context);
}
