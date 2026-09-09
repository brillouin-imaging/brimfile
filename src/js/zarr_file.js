import * as zarr from "https://cdn.jsdelivr.net/npm/zarrita/+esm";
//Imports for reading zip files
import ZipStore from "https://cdn.jsdelivr.net/npm/@zarrita/storage/zip/+esm";
//imports for reading zarr files from S3 buckets
import FetchStore from "https://cdn.jsdelivr.net/npm/@zarrita/storage/fetch/+esm";
import { XMLParser } from "https://cdn.jsdelivr.net/npm/fast-xml-parser/+esm";

// This function is used to standardize the path by ensuring it ends with a '/' and does not start with one.
function standardize_path(path) {
  if (!path.endsWith('/')) {
    path = path + '/';
  }
  if (path.startsWith('/')) {
    path = path.slice(1);
  }
  return path;
}


class FolderStore {
  constructor(fileMap) { 
    this.fileMap = fileMap;
  }

  // Read file bytes
  async get(key) {
    key = key.replace(/^\/+/, ""); // remove leading slash
    const file = this.fileMap.get(key);
    if (!file) return undefined;
    return new Uint8Array(await file.arrayBuffer());
  }

  // Confirm if file or folder exists
  async has(key) {
    key = key.replace(/^\/+/, "");
    // directories should have a trailing slash
    if (this.fileMap.has(key)) return true;
    if (!key.endsWith("/")) {
      return this.fileMap.has(key + "/");
    }
    return false;
  }

  // List all keys
  async list() {
    const keys = new Set();

    for (const key of this.fileMap.keys()) {
      if (key === "") continue;  

      if (key.endsWith("/")) { 
        keys.add(key);
      } else {
        keys.add(key);

        // Ensure parent directories are included with trailing slash
        const parts = key.split("/");
        for (let i = 1; i < parts.length; i++) {
          const parent = parts.slice(0, i).join("/") + "/";
          keys.add(parent);
        }
      }
    }

    return Array.from(keys);
  }
}

class ZarrFile {
  static StoreType = Object.freeze({
    ZIP: 'zip',
    ZARR: 'zarr',
    S3: 'S3',
    FOLDER: 'folder', 
    AUTO: 'auto'
  })
  constructor() {
    this.ready = false
  }
  is_ready() {
    return this.ready;
  };
  async #wait_for_ready(timeout_ms = 5000) {
    async function sleep(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }
    const start = performance.now();
    while (true) {
      if (this.is_ready()) {
        return;
      }
      else {
        await sleep(1); 
      }
      if (performance.now()-start>timeout_ms) {
        throw new Error(`The zarr file was not ready within the timout of ${timeout_ms}ms`);
      }
    }
  }
  async init(file) {
    this.filename = file.name;
    this.store_type = ZarrFile.StoreType.ZIP;  

    this.store = await ZipStore.fromBlob(file);
    this.root = await zarr.open.v3(this.store, { kind: "group" });  
  }

  async init_from_url(url) {
    this.filename = url;
    this.store_type = ZarrFile.StoreType.S3;

    this.store = new FetchStore(url);
    this.root = await zarr.open.v3(this.store, { kind: "group" });
  }
 

 async init_from_folder(files) {
  const fileMap = new Map();
  let rootName = "";

  for (const file of files) {
    const fullPathRaw = file?.webkitRelativePath || file?.relativePath || file?.name || "";
    // Remove leading slashes to ensure consistent path formatting
    const fullPath = String(fullPathRaw).replace(/^\/+/, "");

    if (!fullPath) {
      throw new Error("Folder files must include a valid relative path");
    }

    if (!rootName) {
      rootName = fullPath.split("/")[0];
    }

    const relativePath = fullPath.split("/").slice(1).join("/") || fullPath;

    fileMap.set(relativePath, file);

    // register directory hierarchy
    const parts = relativePath.split("/");
    for (let i = 1; i < parts.length; i++) {
      const dir = parts.slice(0, i).join("/") + "/";
      if (!fileMap.has(dir)) {
        fileMap.set(dir, null);   
      }
    }
  }

  if (fileMap.size === 0) {
    throw new Error("No files were found in the selected folder");
  }

  this.filename = rootName;
  this.store_type = ZarrFile.StoreType.FOLDER;
  this.store = new FolderStore(fileMap);
  this.root = await zarr.open.v3(this.store, { kind: "group" });
}


  /******** Attribute Management ********/
  async get_attribute(full_path, attr_name) {
    await this.#wait_for_ready()
    full_path = standardize_path(full_path);
    const obj = await zarr.open.v3(this.root.resolve(full_path));

    if (!Object.hasOwn(obj.attrs, attr_name)) {
      throw new Error(
        `Attribute '${attr_name}' does not exist on object '${full_path}'`
      );
    }
    return obj.attrs[attr_name];
  }

  /******** Group Management ********/

  async open_group(full_path){
    await this.#wait_for_ready()
    full_path = standardize_path(full_path);
    if (! await this.object_exists(full_path)) {
      throw new Error(`The object '${full_path}' doesn't exist!`)
    }
    // for now simply return the path since all the functions that accepts a group object require its full path
    return full_path
    //Here is the actual code that would open the group 
    /*
    const group = await zarr.open(this.root.resolve(full_path), { kind: "group" });
    return group
    */
  }

  /******** Dataset Management ********/

  async open_dataset(full_path){
    await this.#wait_for_ready()
    full_path = standardize_path(full_path);
    if (! await this.object_exists(full_path)) {
      throw new Error(`The object '${full_path}' doesn't exist!`)
    }
    // for now simply return the path since the array object itself is not json transferrable
    return full_path
    /*
    const arr = await zarr.open.v3(this.root.resolve(full_path), { kind: "array" });
    return arr
    */
  }

  async get_array_slice(full_path, indices){
    await this.#wait_for_ready()
    const array = await zarr.open.v3(this.root.resolve(full_path), { kind: "array" });
    function undef2null(obj) {return obj===undefined?null:obj;}
    let js_indices = [];
    for (let i of indices) {
      js_indices.push(zarr.slice(undef2null(i[0]), undef2null(i[1])))
    }
    const res = await zarr.get(array, js_indices);
    return res
  }

  async get_array_shape(full_path){
    await this.#wait_for_ready()
    const array = await zarr.open.v3(this.root.resolve(full_path), { kind: "array" });
    return array.shape;
  }

  async get_array_dtype(full_path){
    await this.#wait_for_ready()
    const array = await zarr.open.v3(this.root.resolve(full_path), { kind: "array" });
    return array.dtype;
  }

  /******** Listing ********/

  async #list_S3keys(full_path){

    function split_path(url, full_path) {
      url = standardize_path(url).slice(0,-1);
      full_path = standardize_path(full_path);

      let path = [];
      const last_slash = url.lastIndexOf('/');
      path.endpoint = url.slice(0, last_slash+1)
      path.object = url.slice(last_slash+1) + '/' + full_path
      return path;
    }
    const path = split_path(this.filename, full_path);

    let queries = "list-type=2&delimiter=/";
    queries += "&prefix="+path.object;

    let url = path.endpoint + "?" + queries;
    url = encodeURI(url);

    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
    const xmlText = await response.text();

    // Parse XML using fast-xml-parser
    const parser = new XMLParser();
    const xmlObj = parser.parse(xmlText);

    function ExtractKeyFromPrefix (x) {
        let p = x.Prefix;
        if (p.endsWith('/')) {
            p = p.slice(0, -1)
        }
        p = p.split('/').pop()
        return p;
    }
    // Extract CommonPrefixes
    let prefixes = [];
    if (xmlObj.ListBucketResult && xmlObj.ListBucketResult.CommonPrefixes) {
        const cp = xmlObj.ListBucketResult.CommonPrefixes;
        if (Array.isArray(cp)) {
            prefixes = cp.map(ExtractKeyFromPrefix);
        } else if (cp.Prefix) {
            prefixes = [ExtractKeyFromPrefix(cp)];
        }
    }
    return prefixes;
  }
  async list_objects(full_path) {
    await this.#wait_for_ready()
    const objects = [];
    full_path = standardize_path(full_path);

    if (this.store_type == ZarrFile.StoreType.ZIP) {
      const entries = (await this.store.info).entries;
      const all_objs = Object.keys(entries)
      for (const key of all_objs) {
        //console.log(key);
        if (key.startsWith(full_path)) {
          let obj = key.slice(full_path.length);
          obj = obj.split("/")[0];
          if (!obj.endsWith('zarr.json') && !objects.includes(obj) && obj!=="") {
            //check if it is a valid zarr object
            if (all_objs.includes(full_path+obj+"/zarr.json")) {
              objects.push(obj);
              }
          }
        } 
      }
    }
    else if (this.store_type == ZarrFile.StoreType.S3) {
      return this.#list_S3keys(full_path);
    }
    else if (this.store_type === ZarrFile.StoreType.FOLDER) {
      const all_objs = await this.store.list();
      for (const key of all_objs) {
        if (key.startsWith(full_path)) {
          let obj = key.slice(full_path.length);
          obj = obj.split("/")[0];
          if (!obj.endsWith('zarr.json') && !objects.includes(obj) && obj!=="") {
            //check if it is a valid zarr object
            if (all_objs.includes(full_path+obj+"/zarr.json")) {
              objects.push(obj);
            }
          }
        }
      }
    }
    else {
      throw new Error(this.store_type + ' is not supported!')
    }
    return objects;
  }

  async object_exists(full_path) {
    await this.#wait_for_ready()
    full_path = standardize_path(full_path);
    try {
      await zarr.open.v3(this.root.resolve(full_path));
      return true;
    }
    catch (e) {
      return false;
    }
  }

  async isArray(full_path) {
    try {
        await zarr.open.v3(this.root.resolve(full_path), { kind: "array" });
        return true;
    } catch (error) {
        return false;
    }
  }
  async isGroup(full_path) {
    try {
        await zarr.open.v3(this.root.resolve(full_path), { kind: "group" });
        return true;
    } catch (error) {
        return false;
    }
  }

  async list_attributes(full_path) {
    await this.#wait_for_ready()
    full_path = standardize_path(full_path);
    const obj = await zarr.open.v3(this.root.resolve(full_path));
    return Object.keys(obj.attrs);
  }

  /******** Generating JSON descriptor ********/

  async generateJsonDescriptor() {
    // Recursive function to traverse zarr nodes
    async function recurseNodes(zarr_file, path = '') {
        const nodeDict = {};
        
        // Get attributes
        nodeDict.attributes = {};
        const attrs = await zarr_file.list_attributes(path);
        for (const attr of attrs) {
            nodeDict.attributes[attr] = await zarr_file.get_attribute(path, attr);
        }
        
        // Check if it's a group or array
        if (await zarr_file.isGroup(path)) {
            // It's a group
            nodeDict.node_type = 'group';
            
            // Recursively process children
            const childs = await zarr_file.list_objects(path);
            for (const child of childs) {
                nodeDict[child] = await recurseNodes(zarr_file, path + '/' + child);
            }
        } else {
            // It's an array
            nodeDict.node_type = 'array';
            nodeDict.shape = await zarr_file.get_array_shape(path);
            nodeDict.dtype = await zarr_file.get_array_dtype(path);
        }
        
        return nodeDict;
    }
    
    // Generate the digest starting from root
    const outDict = await recurseNodes(this);
    
    // Convert to JSON and log it
    const jsonOutput = JSON.stringify(outDict, null, 4);
    
    return jsonOutput;
  }
}

/**
 * Initializes a Zarr source from a ZIP file, a URL, or a folder file list.
 *
 * This function starts async initialization and returns immediately. The returned
 * `zarr_file_js` instance is usable once `zarr_file_js.is_ready()` is `true`.
 *
 * @param {Array<File|string>|FileList} files Input source container. Supported forms:
 * - `[File]` where `File.name` ends with `.zip`
 * - `[string]` containing a URL/path to a Zarr root
 * - folder-style file lists (`FileList`/array-like of `File` objects)
 * @returns {{zarr_file_js: ZarrFile, filename: string}} Initialized wrapper object.
 * @throws {Error} If `files` is not one of the supported input forms.
 */
function init_file(files) {
  let zarr_file = new ZarrFile();
  let filename = ""

  if (files.length === 1 && files[0] instanceof File && files[0].name.endsWith('.zip')) {
    filename = files[0].name;
    zarr_file.init(files[0]).then(() => {
      zarr_file.ready = true;
    });
  }
  else if (files.length === 1 && typeof files[0] == 'string') {
    const file = standardize_path(files[0]);
    filename = file;
    //make sure the filename doesn't end with '/'
    if (filename.endsWith('/')) {
      filename = filename.slice(0, -1);
    }
    zarr_file.init_from_url(file).then(() => {
      zarr_file.ready = true;
    });
  }
  else if(
    files
    && typeof files.length === "number"
    && files.length > 0
    && typeof files[0]?.arrayBuffer === "function"
  )
{
  zarr_file.init_from_folder(files).then(() => {
    zarr_file.ready = true;
  });

}
  else {
    throw new Error("'file' needs to be either a File object, a folder file list, or a url")
  }
  return {zarr_file_js: zarr_file, filename: filename};

}


export { ZarrFile, init_file}; 