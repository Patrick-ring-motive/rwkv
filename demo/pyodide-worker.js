/*<![CDATA[*/

importScripts("https://cdn.jsdelivr.net/pyodide/v0.21.0a2/full/pyodide.js");

let pyodide;

self.onmessage = async ({
  data
}) => {
  const {
    type,
    id
  } = data;

  if (type === "init") {
    const {
      baseUrl
    } = data;
    pyodide = await loadPyodide();
    await pyodide.loadPackage("micropip");

    const tokJson = await fetch(baseUrl + "tokenizer.json").then(r => r.text());
    pyodide.FS.writeFile("/tokenizer.json", tokJson, {
      encoding: "utf8"
    });
    pyodide.globals.set("_whl_url", baseUrl + "tokenizers_python-0.11.0-cp310-cp310-emscripten_3_1_14_wasm32.whl");

    await pyodide.runPythonAsync(`
import os, micropip
os.environ["TOKENIZERS_PARALLELISM"] = "0"
await micropip.install(_whl_url)
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("/tokenizer.json")
    `);

    self.postMessage({
      type: "ready"
    });
    return;
  }

  if (type === "encode") {
    pyodide.globals.set("_in", data.text);
    pyodide.runPython(`_ids = tokenizer.encode(_in).ids`);
    self.postMessage({
      type: "encoded",
      id,
      tokens: Array.from(pyodide.globals.get("_ids").toJs()),
    });
    return;
  }

  if (type === "decode") {
    // data.tokens is an array of one or more token ids (batch decode supported)
    pyodide.globals.set("_t", data.tokens.join(","));
    pyodide.runPython(`_dec = tokenizer.decode([int(x) for x in _t.split(",")])`);
    self.postMessage({
      type: "decoded",
      id,
      text: pyodide.globals.get("_dec")
    });
  }
};

/*]]>*/
