import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import test from 'node:test';

import { resolve } from './loader.mjs';

registerHooks({ resolve });
const { init_file, ZarrFile } = await import('../../src/js/zarr_file.js');

function folderFile(path, attributes = {}) {
  const file = new File(
    [JSON.stringify({ zarr_format: 3, node_type: 'group', attributes })],
    path.split('/').at(-1),
  );
  Object.defineProperty(file, 'webkitRelativePath', { value: path });
  return file;
}

test('folder name is returned immediately and the store remains usable', async () => {
  const files = [
    folderFile('/sample.brim/zarr.json', { label: 'root' }),
    folderFile('/sample.brim/child/zarr.json', { label: 'child' }),
  ];
  const result = init_file(files);

  assert.equal(result.filename, 'sample.brim');
  assert.equal(result.zarr_file_js.filename, result.filename);
  assert.equal(result.zarr_file_js.is_ready(), false);
  assert.equal(await result.zarr_file_js.get_attribute('', 'label'), 'root');
  assert.deepEqual(await result.zarr_file_js.list_objects(''), ['child']);
  assert.equal(await result.zarr_file_js.get_attribute('child', 'label'), 'child');
});

test('a single folder file and relativePath fallback return the root name', async () => {
  const file = folderFile('', { label: 'root' });
  Object.defineProperty(file, 'relativePath', { value: 'single.brim/zarr.json' });

  const result = init_file([file]);

  assert.equal(result.filename, 'single.brim');
  assert.equal(await result.zarr_file_js.get_attribute('', 'label'), 'root');
});

test('ZIP and URL inputs retain their existing names and dispatch', async (t) => {
  const zipInit = t.mock.method(ZarrFile.prototype, 'init', async () => {});
  const urlInit = t.mock.method(ZarrFile.prototype, 'init_from_url', async () => {});
  const zip = new File([], 'sample.zip');

  const zipResult = init_file([zip]);
  const urlResult = init_file(['https://example.com/sample.brim']);

  assert.equal(zipResult.filename, 'sample.zip');
  assert.equal(urlResult.filename, 'https://example.com/sample.brim');
  assert.equal(zipInit.mock.calls[0].arguments[0], zip);
  assert.equal(urlInit.mock.calls[0].arguments[0], 'https://example.com/sample.brim/');
  await Promise.resolve();
  assert.equal(zipResult.zarr_file_js.is_ready(), true);
  assert.equal(urlResult.zarr_file_js.is_ready(), true);
});
