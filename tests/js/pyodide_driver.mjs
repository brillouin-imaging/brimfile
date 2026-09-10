// Mechanical RPC driver: loads a real pyodide runtime in Node, mounts the
// brimfile source tree into it, wraps a real `ZarrFile` (from
// src/js/zarr_file.js) with the pyodide `_zarrFile` implementation, and
// executes a list of operations against it. All assertions live in the
// Python test suite (tests/test_file_abstraction_pyodide.py); this script
// only executes operations and reports raw results/errors as JSON.
//
// Usage: node pyodide_driver.mjs <path-to-zarr_file.js> <path-to-repo-src>
// Input (stdin): {"base_url": "...", "ops": [{"method": ..., "args": [...], "post": ...}, ...]}
// Output (stdout): {"results": [{"ok": true, "value": ...} | {"ok": false, "error": "..."}]}

import { register } from 'node:module';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

register('./loader.mjs', import.meta.url);

const [zarrFileJsPath, repoSrcPath] = process.argv.slice(2);
if (!zarrFileJsPath || !repoSrcPath) {
  console.error('Usage: node pyodide_driver.mjs <path-to-zarr_file.js> <path-to-repo-src>');
  process.exit(2);
}

const { loadPyodide } = await import('pyodide');
const { ZarrFile } = await import(pathToFileURL(zarrFileJsPath).href);

async function main() {
  const input = JSON.parse(readFileSync(0, 'utf-8'));
  const { base_url: baseUrl, ops } = input;

  const zarrFileJs = new ZarrFile();
  await zarrFileJs.init_from_url(baseUrl);
  // init_from_url() itself doesn't flip the ready flag (only the init_file()
  // wrapper in zarr_file.js does) so we set it directly, as init_file() would.
  zarrFileJs.ready = true;

  const pyodide = await loadPyodide({
    // keep stdout clean for our own JSON-on-stdout protocol below
    stdout: (msg) => process.stderr.write(msg + '\n'),
    stderr: (msg) => process.stderr.write(msg + '\n'),
  });
  await pyodide.loadPackage('numpy');
  pyodide.mountNodeFS('/repo_src', repoSrcPath);
  pyodide.globals.set('zarr_js', zarrFileJs);
  pyodide.globals.set('ops_json', JSON.stringify(ops));

  const resultsJson = await pyodide.runPythonAsync(`
import sys
if '/repo_src' not in sys.path:
    sys.path.insert(0, '/repo_src')

import json
import numpy as np
from brimfile.file_abstraction import _AbstractFile, _async_getitem

_f = _AbstractFile(zarr_js, filename="pyodide_test", version=None)

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

  process.stdout.write('<<<RESULT>>>\n' + resultsJson);
}

main().catch((err) => {
  process.stderr.write(String((err && err.stack) || err) + '\n');
  process.exit(1);
});
