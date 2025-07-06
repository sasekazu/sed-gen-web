// main.js

/* WASM バイナリの設定 - CDNから読み込み */
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';

/* デバッグのため詳細ログを有効化 */
ort.env.logLevel = 'verbose';

/* 定数 */
const MODEL_URL = './unet.onnx';
const SIZE      = 224;

/* DOM 取得 */
const fileInput    = document.getElementById('file');
const inputCanvas  = document.getElementById('inputCanvas');
const outputCanvas = document.getElementById('outputCanvas');
const inCtx  = inputCanvas.getContext('2d');
const outCtx = outputCanvas.getContext('2d');

/* セッション生成（1 回だけ） - WASMのみを使用 */
const sessionPromise = ort.InferenceSession.create(MODEL_URL, {
  executionProviders: ['wasm']
}).catch(error => {
  console.error('Failed to create ONNX session:', error);
  throw error;
});

/* 画像アップロード */
fileInput.addEventListener('change', async ({ target }) => {
  try {
    const file = target.files?.[0];
    if (!file) return;

    console.log('Loading ONNX session...');
    const session = await sessionPromise;
    console.log('ONNX session loaded successfully');
    
    const img = await fileToImage(file);

    /* 入力画像 224×224 にリサイズ描画 */
    inCtx.clearRect(0, 0, SIZE, SIZE);
    inCtx.drawImage(img, 0, 0, SIZE, SIZE);

    /* RGBA → BT.601 グレースケール → Float32Array(0–1) */
    const rgba = inCtx.getImageData(0, 0, SIZE, SIZE).data;
    const gray = new Float32Array(SIZE * SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      const p = i * 4;
      gray[i] = (0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2]) / 255.0;
    }
    const input = new ort.Tensor('float32', gray, [1, 1, SIZE, SIZE]);

    /* 推論 */
    const { output } = await session.run({ input });

    /* モデル出力をそのままグレースケールとして描画 */
    const mask = new ImageData(SIZE, SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      const gray = Math.round(Math.max(0, Math.min(255, output.data[i] * 255))); // 0-255の範囲にクランプ
      const p = i * 4;
      mask.data[p]     = gray;  // R
      mask.data[p + 1] = gray;  // G
      mask.data[p + 2] = gray;  // B
      mask.data[p + 3] = 255;   // A (完全不透明)
    }
    outCtx.putImageData(mask, 0, 0);
  } catch (error) {
    console.error('Error during inference:', error);
  }
});

/* File → Image 変換 */
function fileToImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload  = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = (e) => { URL.revokeObjectURL(url); reject(e); };
    img.src = url;
  });
}
