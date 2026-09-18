// Mechanical RPC driver for the Pyodide branch of `_zarrFile`, rewritten for
// the "real zarr everywhere" design: this loads a real pyodide runtime in
// Node, installs the real `zarr` package (NOT a micropip mock package -
// `zarr` and `numcodecs` are Pyodide-bundled packages as of Pyodide 314.x;
// see the preflight check below for the version caveat on the older 0.29.x
// line this repo/BrimView currently pin), mounts the repo's `src/` directory
// into it, builds small JS objects that duck-type just enough of a browser
// `File`/FileList for `_zarrFile`'s Pyodide branch (`.size`, `.slice(start,
// stop).arrayBuffer()`, `.name`/`.webkitRelativePath`, `.length` + indexing),
// and executes a list of operations against a real `brimfile.File`. All
// assertions live in the Python test suite
// (tests/test_file_abstraction_pyodide.py); this script only executes
// operations and reports raw results/errors as JSON.
//
// Usage: node pyodide_driver.mjs <source-kind> <path-to-repo-src> <path-to-fixture-root-or-url>
//   source-kind: "zip" | "folder" | "url"
//   For "zip"/"folder", <path-to-fixture-root-or-url> is a local directory
//   (e.g. a `simple_brim_file` .zarr directory); for "url" it's a full URL.
// Input (stdin): {"ops": [{"method": ..., "args": [...], "post": ...}, ...]}
// Output (stdout): a line starting with "<<<r>>>" followed by {"results": [...]}
//   or, if the `zarr`/`numcodecs` preflight fails, a line starting with
//   "<<<preflight_error>>>" followed by a short diagnostic message.

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative, basename } from 'node:path';

const [sourceKind, repoSrcPath, fixtureRootOrUrl] = process.argv.slice(2);
if (!sourceKind || !repoSrcPath || !fixtureRootOrUrl) {
  console.error('Usage: node pyodide_driver.mjs <zip|folder|url> <path-to-repo-src> <fixture-root-or-url>');
  process.exit(2);
}

const { loadPyodide } = await import('pyodide');

// -- minimal JsProxy-shaped stand-ins for a browser File/FileList --------

class FakeJsFile {
  constructor(name, data, webkitRelativePath) {
    this.name = name;
    this.size = data.length;
    this._data = data;
    if (webkitRelativePath !== undefined) this.webkitRelativePath = webkitRelativePath;
  }
  slice(start, stop) {
    return new FakeJsBlob(this._data.subarray(start, stop));
  }
}

class FakeJsBlob {
  constructor(data) { this._data = data; }
  async arrayBuffer() {
    return this._data.buffer.slice(this._data.byteOffset, this._data.byteOffset + this._data.byteLength);
  }
}

function walkFiles(rootDir) {
  const out = [];
  const rec = (dir) => {
    for (const name of readdirSync(dir)) {
      const full = join(dir, name);
      if (statSync(full).isDirectory()) rec(full);
      else out.push(full);
    }
  };
  rec(rootDir);
  return out;
}

function buildFakeFileArray(rootDir) {
  const rootName = basename(rootDir);
  return walkFiles(rootDir).map((full) => {
    const rel = join(rootName, relative(rootDir, full)).split('\\').join('/');
    return new FakeJsFile(basename(full), readFileSync(full), rel);
  });
}

async function main() {
  const input = JSON.parse(readFileSync(0, 'utf-8'));
  const { ops } = input;

  const pyodide = await loadPyodide({
    stdout: (msg) => process.stderr.write(msg + '\n'),
    stderr: (msg) => process.stderr.write(msg + '\n'),
  });

  // --- preflight: real zarr/numcodecs availability on this pyodide build ---
  // Confirmed directly against https://pyodide.org/en/<version>/usage/packages-in-pyodide.html:
  // zarr 3.x (paired with a compatible numcodecs>=0.14) is only bundled
  // starting at Pyodide 314.0.1 (it was briefly missing from 314.0.0 itself
  // due to a build issue, then fixed 17 days later). Every 0.29.x release
  // either has no zarr at all, or (0.27.x and earlier) only the old zarr 2.x.
  if (pyodide.version === '314.0.0' || /^0\./.test(pyodide.version)) {
    process.stdout.write('<<<preflight_error>>>\n' +
      `This driver requires Pyodide >= 314.0.1 for real zarr 3.x support ` +
      `(currently running ${pyodide.version}). zarr 3.x + a compatible ` +
      `numcodecs>=0.14 are bundled starting at Pyodide 314.0.1; every ` +
      `0.29.x release (see tests/js/package.json) either has no zarr at ` +
      `all or only the incompatible old zarr 2.x, and the single 314.0.0 ` +
      `release briefly dropped zarr due to a build issue. Bump the ` +
      `"pyodide" devDependency in tests/js/package.json (and, for a real ` +
      `deployment, BrimView's own pinned Pyodide version) to 314.0.1 or ` +
      `later.\n`);
    process.exit(0);
  }
  await pyodide.loadPackage(['micropip']);
  const micropip = pyodide.pyimport('micropip');
  try {
    await micropip.install('zarr>=3.1.1');
  } catch (e) {
    process.stdout.write('<<<preflight_error>>>\n' + String(e) + '\n');
    process.exit(0);
  }

  pyodide.mountNodeFS('/repo_src', repoSrcPath);

  if (sourceKind === 'zip') {
    // Build a real zip archive of the fixture directory using pyodide's own
    // (real) stdlib zipfile, then wrap the resulting bytes as a fake browser
    // File - exercising the exact same `_read_whole_js_file` +
    // `zarr.storage.ZipStore(io.BytesIO(...))` path a real browser File would.
    const zipBytesB64 = pyodide.runPython(`
import zipfile, io, os, base64
buf = io.BytesIO()
root_dir = ${JSON.stringify(fixtureRootOrUrl)}
with zipfile.ZipFile(buf, "w") as zf:
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            full = os.path.join(dirpath, name)
            zf.write(full, os.path.relpath(full, root_dir))
base64.b64encode(buf.getvalue()).decode("ascii")
`);
    const zipBytes = Buffer.from(zipBytesB64, 'base64');
    pyodide.globals.set('js_source', new FakeJsFile('sample.zip', zipBytes));
    pyodide.globals.set('store_type_name', 'ZIP');
  } else if (sourceKind === 'folder') {
    const files = buildFakeFileArray(fixtureRootOrUrl);
    pyodide.globals.set('js_source', files);
    pyodide.globals.set('store_type_name', 'FOLDER');
  } else if (sourceKind === 'url') {
    pyodide.globals.set('js_source', fixtureRootOrUrl);
    pyodide.globals.set('store_type_name', 'S3');
  } else {
    throw new Error(`unknown source kind: ${sourceKind}`);
  }

  pyodide.globals.set('ops_json', JSON.stringify(ops));

  const resultsJson = await pyodide.runPythonAsync(`
import sys
if '/repo_src' not in sys.path:
    sys.path.insert(0, '/repo_src')

import json
import numpy as np
from brimfile.file_abstraction import _AbstractFile, StoreType, _async_getitem

_f = _AbstractFile(js_source, store_type=StoreType[store_type_name])

async def _run_ops(ops):
    results = []
    for op in ops:
        method = op['method']
        args = op.get('args', [])
        post = op.get('post')
        try:
            fn = getattr(_f, method)
            res = fn(*args)
            if hasattr(res, '__await__'):
                res = await res
            if post == 'to_list':
                data = await _async_getitem(res, Ellipsis)
                res = np.asarray(data).tolist()
            elif post == 'shape':
                res = list(res.shape)
            elif post == 'list':
                res = list(res)
            elif post == 'sorted_list':
                res = sorted(list(res))
            results.append({'ok': True, 'value': res})
        except Exception as e:
            results.append({'ok': False, 'error': f'{type(e).__name__}: {e}'})
    return results

results = await _run_ops(json.loads(ops_json))
json.dumps(results)
`);

  process.stdout.write('<<<r>>>\n' + resultsJson);
}

main().catch((err) => {
  process.stderr.write(String((err && err.stack) || err) + '\n');
  process.exit(1);
});
