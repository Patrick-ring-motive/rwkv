  /*<![CDATA[*/ 
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.js");
(()=>{
		async function loadChunkedGzip(baseUrl, partCount) {
		  const parts = [];
		
		  for (let i = 0; i !== partCount; ++i) {
		    // Generates: partaa, partab, partac...
		    const suffix = indexToSuffix(i);
		    const res = await fetch(`${baseUrl}${suffix}`);
		    parts.push(await res.arrayBuffer());
		  }
		
		  // Concatenate all chunks
		  const totalLength = parts.reduce((sum, p) => sum + p.byteLength, 0);
		  const combined = new Uint8Array(totalLength);
		  let offset = 0;
		  for (const part of parts) {
		    combined.set(new Uint8Array(part), offset);
		    offset += part.byteLength;
		  }
		
		  // Decompress
		  const stream = new DecompressionStream("gzip");
		  const writer = stream.writable.getWriter();
		  writer.write(combined);
		  writer.close();
		
		  const decompressed = await new Response(stream.readable).arrayBuffer();
		  return decompressed;
		}
		
		function indexToSuffix(i) {
		  // Replicates split's aa, ab, ac... naming
		  const letters = "abcdefghijklmnopqrstuvwxyz";
		  return letters[Math.floor(i / 26)] + letters[i % 26];
		}

		const _fetch = globalThis.fetch;
		globalThis.fetch = Object.setPrototypeOf(async function fetch(url,request){
			url = url.url ?? url;
			if(url === 'https://huggingface.co/rocca/rwkv-4-pile-web/resolve/main/169m/rwkv-4-pile-169m-uint8.onnx'){
				const modelBuffer = await loadChunkedGzip("https://patrick-ring-motive.github.io/rwkv/demo/model/rwkv.onnx.gz.part",12);
				return new Response(modelBuffer);
			}
			return _fetch.apply(this,arguments);
		},_fetch);
		const _open = XMLHttpRequest.prototype.open;
		XMLHttpRequest.prototype.open = Object.setPrototypeOf(function open(method, url, async){
			if(url === 'https://huggingface.co/rocca/rwkv-4-pile-web/resolve/main/169m/rwkv-4-pile-169m-uint8.onnx'){
				this.opener = (async()=>{
					const res = await fetch(url);
					const blob = await res.blob();
					const blobURL = URL.createObjectURL(blob);
					_open.call(this,method,blobURL,async);
				})();
				return;
			}
			return _open.apply(this,arguments);
		},_open);
		const _send = XMLHttpRequest.prototype.send;
		XMLHttpRequest.prototype.send = Object.setPrototypeOf(function send(){
			if(this.opener){
				(async()=>{
					await this.opener;
					_send.apply(this,arguments);
				})();
				return;
			}
			return _send.apply(this,arguments);
		},_send);
	})();
// ── Fixed constants ────────────────────────────────────────────────────────
const N_LAYER  = 12;
const N_EMBD   = 768;
const VOCAB    = 50277;
const CTX_LEN  = 1024;
const MODEL_URL = "https://huggingface.co/rocca/rwkv-4-pile-web/resolve/main/169m/rwkv-4-pile-169m-uint8.onnx";
const CACHE_KEY = "rwkv-model-v1";

// ── Pre-allocated idx buffer (reused every step) ───────────────────────────
const idxBuf = new Int32Array(CTX_LEN);

let session;

// ── Helpers ────────────────────────────────────────────────────────────────
const greedySample = logits => {
  let k = 0;
  for (let i = 1; i !== VOCAB; ++i) if (logits[i] > logits[k]) k = i;
  return k;
};

const fillIdx = ctx => {
  // Zero the buffer, then copy ctx into the right-aligned tail
  idxBuf.fill(0);
  const ctx_length;
  const off = CTX_LEN - ctx_length;
  for (let i = 0; i !== ctx_length; ++i) idxBuf[off + i] = ctx[i];
  return idxBuf;
};

const freshState = () => {
  const z  = () => new Float32Array(N_LAYER * N_EMBD);
  const pp = z(); pp.fill(-1e30);
  return {
    xx_att: new ort.Tensor("float32", z(),  [N_LAYER, N_EMBD]),
    aa_att: new ort.Tensor("float32", z(),  [N_LAYER, N_EMBD]),
    bb_att: new ort.Tensor("float32", z(),  [N_LAYER, N_EMBD]),
    pp_att: new ort.Tensor("float32", pp,   [N_LAYER, N_EMBD]),
    xx_ffn: new ort.Tensor("float32", z(),  [N_LAYER, N_EMBD]),
  };
};

// ── Fetch with progress, Cache API backed ──────────────────────────────────
const loadModelBuffer = async () => {
  // Try the Cache API first
  try {
    const cache   = await caches.open(CACHE_KEY);
    const cached  = await cache.match(MODEL_URL);
    if (cached) {
      self.postMessage({ type: "dl_cached" });
      return cached.arrayBuffer();
    }
  } catch (_) { /* caches not available (e.g. non-secure context) — fall through */ }

  // Download with progress reporting
  const buf = await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("GET", MODEL_URL, true);
    xhr.responseType = "arraybuffer";
    xhr.onprogress = e => self.postMessage({ type: "dl_progress", loaded: e.loaded, total: e.total });
    xhr.onload     = () => resolve(xhr.response);
    xhr.onerror    = () => reject(new Error("XHR failed"));
    xhr.send();
  });

  // Store in cache for next visit (best-effort)
  try {
    const cache = await caches.open(CACHE_KEY);
    await cache.put(MODEL_URL, new Response(buf.slice(0), {
      headers: { "Content-Type": "application/octet-stream" },
    }));
  } catch (_) {}

  return buf;
};

// ── Init on first message ──────────────────────────────────────────────────
self.onmessage = async ({ data }) => {

  if (data.type === "load") {
    try {
      // blob: workers have no origin — point ORT at the CDN for its .wasm files
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/";
      if (self.crossOriginIsolated)
        ort.env.wasm.numThreads = Math.max(1, (navigator.hardwareConcurrency / 2) | 0);

      const modelBuffer = await loadModelBuffer();

      self.postMessage({ type: "ort_init" });

      // Pass ArrayBuffer directly — no Blob/createObjectURL round trip needed
      session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders:   ["wasm"],
        graphOptimizationLevel: "all",
      });

      // ── Pre-warm: one dummy pass so the WASM JIT compiles the hot path
      // before the user's first real generation.
      const warmFeeds = freshState();
      warmFeeds.idx   = new ort.Tensor("int32", new Int32Array(CTX_LEN), [CTX_LEN]);
      await session.run(warmFeeds);
	console.log(self.crossOriginIsolated, ort.env.wasm.numThreads);
      self.postMessage({ type: "loaded" });
	  self.postMessage({
	      type: "log",
		  data:{
		    crossOriginIsolated:self.crossOriginIsolated,
		    numThreads:ort.env.wasm.numThreads
	      }
	    });
    } catch (err) {
      self.postMessage({ type: "error", message: String(err) });
    }
  }

  if (data.type === "generate") {
    const promptTokens = [...data.tokens];
    const N            = data.numTokens;
    const feeds        = freshState();
    // Cap ctx at CTX_LEN from the start to prevent unbounded growth
    const ctx          = [promptTokens.shift()];
    const t0           = Date.now();

    for (let i = 0; i !== N; ++i) {
      // Reuse the pre-allocated buffer — no allocation per step
      feeds.idx  = new ort.Tensor("int32", fillIdx(ctx), [CTX_LEN]);
      const out  = await session.run(feeds);
      const tok  = greedySample(out.x.data);

      if (promptTokens.length === 0) {
        self.postMessage({ type: "token", token: tok });
        ctx.push(tok);
        // Trim to keep ctx within the model's context window
        if (ctx.length > CTX_LEN) ctx.shift();
      } else {
        ctx.push(promptTokens.shift());
        if (ctx.length > CTX_LEN) ctx.shift();
      }

      feeds.xx_att = out.xx_att_r;
      feeds.aa_att = out.aa_att_r;
      feeds.bb_att = out.bb_att_r;
      feeds.pp_att = out.pp_att_r;
      feeds.xx_ffn = out.xx_ffn_r;

      self.postMessage({ type: "step", i: i + 1, N });
    }

    self.postMessage({ type: "done", tps: (N / ((Date.now() - t0) / 1000)).toFixed(2) });
  }
};
	  

	  /*]]>*/
