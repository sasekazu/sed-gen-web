// main.js

/* WASM バイナリの設定 - CDNから読み込み */
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';

/* デバッグのため詳細ログを有効化 */
ort.env.logLevel = 'verbose';

/* 定数 */
const MODEL_URL = './unet.onnx';
const SIZE      = 224;
const PRESSURE_MASK_URL = './resources/mask/mask-pressure.png';
const SED_MASK_URL = './resources/mask/mask-sed.png';
const DEFAULT_IMAGE_URL = './resources/input/file01_00050.png';

/* 画像リストの定義 */
const INPUT_IMAGES = [
  'file01_00050.png', 'file01_00148.png', 'file01_00232.png', 'file01_00337.png', 'file01_00421.png',
  'file01_00526.png', 'file01_00617.png', 'file01_00715.png', 'file01_00918.png', 'file01_00988.png',
  'file02_00134.png', 'file02_00239.png', 'file02_00344.png', 'file02_00428.png', 'file02_00526.png',
  'file02_00617.png', 'file02_00715.png', 'file02_00813.png', 'file02_00904.png', 'file02_00988.png'
];

/* DOM 取得 */
const imageSlider = document.getElementById('imageSlider');
const imageInfo = document.getElementById('imageInfo');
const inferenceTime = document.getElementById('inferenceTime');
const inputCanvas  = document.getElementById('inputCanvas');
const targetCanvas = document.getElementById('targetCanvas');
const outputCanvas = document.getElementById('outputCanvas');
const inCtx  = inputCanvas.getContext('2d');
const targetCtx = targetCanvas.getContext('2d');
const outCtx = outputCanvas.getContext('2d');

/* モード関連のDOM要素 */
const sliderMode = document.getElementById('sliderMode');
const interactiveMode = document.getElementById('interactiveMode');
const sliderContainer = document.getElementById('sliderContainer');
const interactiveContainer = document.getElementById('interactiveContainer');
const targetSection = document.getElementById('targetSection');
const clearButton = document.getElementById('clearButton');

/* グローバル変数 */
let currentMode = 'slider';
let interactiveInputData = new Float32Array(SIZE * SIZE); // Interactive mode用の入力データ
let permanentInputData = new Float32Array(SIZE * SIZE); // マウスボタンが押されている間の永続的な入力データ
let isProcessing = false; // 推論中フラグ
let isMouseDown = false; // マウスボタンが押されているかどうか

/* マスク画像の読み込み（事前に読み込んでおく） */
let pressureMask = null;
let sedMask = null;

/* プリロードされた画像の格納 */
let preloadedImages = [];
let preloadedTargetImages = [];

const loadMasks = async () => {
  try {
    console.log('Loading masks...');
    pressureMask = await loadMaskImage(PRESSURE_MASK_URL);
    console.log('Pressure mask loaded successfully');
    sedMask = await loadMaskImage(SED_MASK_URL);
    console.log('SED mask loaded successfully');
    console.log('All masks loaded successfully');
  } catch (error) {
    console.error('Failed to load masks:', error);
    throw error;
  }
};

/* 全ての入力画像をプリロード */
const preloadInputImages = async () => {
  try {
    console.log('Preloading input images...');
    preloadedImages = await Promise.all(
      INPUT_IMAGES.map(async (filename, index) => {
        try {
          const url = `./resources/input/${filename}`;
          const img = await urlToImage(url);
          console.log(`Loaded input image ${index + 1}/${INPUT_IMAGES.length}: ${filename}`);
          return img;
        } catch (error) {
          console.warn(`Failed to load input image: ${filename}`, error);
          return null;
        }
      })
    );
    console.log('All input images preloaded successfully');
  } catch (error) {
    console.error('Failed to preload input images:', error);
    throw error;
  }
};

/* 全てのターゲット画像をプリロード */
const preloadTargetImages = async () => {
  try {
    console.log('Preloading target images...');
    preloadedTargetImages = await Promise.all(
      INPUT_IMAGES.map(async (filename, index) => {
        try {
          const url = `./resources/target/${filename}`;
          const img = await urlToImage(url);
          console.log(`Loaded target image ${index + 1}/${INPUT_IMAGES.length}: ${filename}`);
          return img;
        } catch (error) {
          console.warn(`Failed to load target image: ${filename}`, error);
          return null;
        }
      })
    );
    console.log('All target images preloaded successfully');
  } catch (error) {
    console.error('Failed to preload target images:', error);
    throw error;
  }
};

/* マスク画像を読み込みCanvasに描画してImageDataを取得 */
function loadMaskImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = SIZE;
      canvas.height = SIZE;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, SIZE, SIZE);
      const imageData = ctx.getImageData(0, 0, SIZE, SIZE);
      resolve(imageData);
    };
    img.onerror = reject;
    img.src = url;
  });
}

/* マスクを適用する関数（しきい値128で黒い部分をマスク化） */
function applyMask(imageData, maskData, backgroundColor = [255, 255, 255]) {
  const result = new ImageData(SIZE, SIZE);
  const data = result.data;
  
  for (let i = 0; i < SIZE * SIZE; i++) {
    const p = i * 4;
    // マスクのグレースケール値を取得（R成分を使用）
    const maskValue = maskData.data[p];
    // しきい値128で黒い部分をマスク化（Pythonコードと同様）
    const isMasked = maskValue < 128;
    
    if (isMasked) {
      // マスクされた部分は元の画像データを使用
      data[p]     = imageData.data[p];     // R
      data[p + 1] = imageData.data[p + 1]; // G
      data[p + 2] = imageData.data[p + 2]; // B
      data[p + 3] = imageData.data[p + 3]; // A
    } else {
      // マスクされていない部分は背景色を使用
      data[p]     = backgroundColor[0]; // R
      data[p + 1] = backgroundColor[1]; // G
      data[p + 2] = backgroundColor[2]; // B
      data[p + 3] = 255;                // A
    }
  }
  
  return result;
}

/* セッション生成（1 回だけ） - WASMのみを使用 */
const sessionPromise = ort.InferenceSession.create(MODEL_URL, {
  executionProviders: ['wasm']
}).catch(error => {
  console.error('Failed to create ONNX session:', error);
  throw error;
});

/* スライダーで画像選択の処理を関数として分離 */
async function onSliderChange(event) {
  try {
    const index = parseInt(event.target.value);
    const img = preloadedImages[index];
    const targetImg = preloadedTargetImages[index];
    
    if (img) {
      // 画像情報を更新
      updateImageInfo(index);
      // 画像を処理
      await processImage(img, targetImg);
    } else {
      console.warn(`Image at index ${index} is not available`);
    }
  } catch (error) {
    console.error('Error during image selection:', error);
  }
}

/* スライダーで画像選択（初期設定では有効） */
imageSlider.addEventListener('input', onSliderChange);

/* 画像情報の更新 */
function updateImageInfo(index) {
  const filename = INPUT_IMAGES[index];
  imageInfo.textContent = `Image ${index + 1} / ${INPUT_IMAGES.length}: ${filename}`;
}

/* 画像処理の共通ロジック */
async function processImage(img, targetImg = null) {
  if (isProcessing) return; // 推論中は処理しない
  
  try {
    isProcessing = true;
    
    // マスクが読み込まれていない場合は読み込む
    if (!pressureMask || !sedMask) {
      console.log('Loading masks...');
      await loadMasks();
    }

    console.log('Loading ONNX session...');
    const session = await sessionPromise;
    console.log('ONNX session loaded successfully');

    /* 入力画像 224×224 にリサイズ描画 */
    inCtx.clearRect(0, 0, SIZE, SIZE);
    inCtx.drawImage(img, 0, 0, SIZE, SIZE);

    /* ターゲット画像の表示 */
    if (targetImg) {
      targetCtx.clearRect(0, 0, SIZE, SIZE);
      targetCtx.drawImage(targetImg, 0, 0, SIZE, SIZE);
      
      /* ターゲット画像をturboカラーマップで変換してからマスクを適用 */
      const targetImageData = targetCtx.getImageData(0, 0, SIZE, SIZE);
      const targetTurboImageData = new ImageData(SIZE, SIZE);
      
      for (let i = 0; i < SIZE * SIZE; i++) {
        const p = i * 4;
        // RGBA → BT.601 グレースケール → 0-1正規化
        const grayValue = (0.299 * targetImageData.data[p] + 0.587 * targetImageData.data[p + 1] + 0.114 * targetImageData.data[p + 2]) / 255.0;
        const [r, g, b] = grayToTurbo(grayValue);
        targetTurboImageData.data[p]     = r;   // R
        targetTurboImageData.data[p + 1] = g;   // G
        targetTurboImageData.data[p + 2] = b;   // B
        targetTurboImageData.data[p + 3] = 255; // A (完全不透明)
      }
      
      const maskedTargetImageData = applyMask(targetTurboImageData, sedMask);
      targetCtx.putImageData(maskedTargetImageData, 0, 0);
    } else {
      // ターゲット画像がない場合はキャンバスをクリア
      targetCtx.clearRect(0, 0, SIZE, SIZE);
    }

    /* 推論用：マスクをかける前の元画像データを取得 */
    const originalImageData = inCtx.getImageData(0, 0, SIZE, SIZE);

    /* RGBA → BT.601 グレースケール → Float32Array(0–1) */
    const rgba = originalImageData.data;
    const gray = new Float32Array(SIZE * SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      const p = i * 4;
      gray[i] = (0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2]) / 255.0;
    }
    const input = new ort.Tensor('float32', gray, [1, 1, SIZE, SIZE]);

    /* 推論（時間計測あり） */
    console.log('Starting inference...');
    const startTime = performance.now();
    const { output } = await session.run({ input });
    const endTime = performance.now();
    const inferenceTimeMs = endTime - startTime;
    
    console.log(`Inference completed in ${inferenceTimeMs.toFixed(2)} ms`);
    
    /* 推論時間を表示に更新 */
    inferenceTime.textContent = `Inference time: ${inferenceTimeMs.toFixed(2)} ms`;

    /* 表示用：入力画像をturboカラーマップで変換してからマスクを適用 */
    const inputTurboImageData = new ImageData(SIZE, SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      const grayValue = gray[i]; // 0-1の正規化済みグレースケール値
      const [r, g, b] = grayToTurbo(grayValue);
      const p = i * 4;
      inputTurboImageData.data[p]     = r;   // R
      inputTurboImageData.data[p + 1] = g;   // G
      inputTurboImageData.data[p + 2] = b;   // B
      inputTurboImageData.data[p + 3] = 255; // A (完全不透明)
    }
    const maskedInputImageData = applyMask(inputTurboImageData, pressureMask);
    inCtx.putImageData(maskedInputImageData, 0, 0);

    /* モデル出力をturboカラーマップで描画 */
    const outputImageData = new ImageData(SIZE, SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      const grayValue = Math.max(0, Math.min(1, output.data[i])); // 0-1の範囲にクランプ
      const [r, g, b] = grayToTurbo(grayValue);
      const p = i * 4;
      outputImageData.data[p]     = r;   // R
      outputImageData.data[p + 1] = g;   // G
      outputImageData.data[p + 2] = b;   // B
      outputImageData.data[p + 3] = 255; // A (完全不透明)
    }

    /* 表示用：出力画像にSEDマスクを適用 */
    const maskedOutputImageData = applyMask(outputImageData, sedMask);
    outCtx.putImageData(maskedOutputImageData, 0, 0);
  } catch (error) {
    console.error('Error during inference:', error);
  } finally {
    isProcessing = false;
  }
}

/* デフォルト画像（最初の画像）を処理 */
async function loadDefaultImage() {
  try {
    if (preloadedImages.length > 0 && preloadedImages[0]) {
      console.log('Processing default image (first preloaded image)...');
      updateImageInfo(0);
      const targetImg = preloadedTargetImages.length > 0 ? preloadedTargetImages[0] : null;
      await processImage(preloadedImages[0], targetImg);
      console.log('Default image processed successfully');
    } else {
      console.warn('No preloaded images available');
    }
  } catch (error) {
    console.error('Error processing default image:', error);
  }
}

/* URL → Image 変換 */
function urlToImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

/* Turbo カラーマップデータ（matplotlib の turbo に相当） */
const TURBO_COLORMAP = [
  [0.18995, 0.07176, 0.23217], [0.19483, 0.08339, 0.26149], [0.19956, 0.09498, 0.29024], [0.20415, 0.10652, 0.31844], [0.20860, 0.11802, 0.34607],
  [0.21291, 0.12947, 0.37314], [0.21708, 0.14087, 0.39964], [0.22111, 0.15223, 0.42558], [0.22500, 0.16354, 0.45096], [0.22875, 0.17481, 0.47578],
  [0.23236, 0.18603, 0.50004], [0.23582, 0.19720, 0.52373], [0.23915, 0.20833, 0.54686], [0.24234, 0.21941, 0.56942], [0.24539, 0.23044, 0.59142],
  [0.24830, 0.24143, 0.61286], [0.25107, 0.25237, 0.63374], [0.25369, 0.26327, 0.65406], [0.25618, 0.27412, 0.67381], [0.25853, 0.28492, 0.69300],
  [0.26074, 0.29568, 0.71162], [0.26280, 0.30639, 0.72968], [0.26473, 0.31706, 0.74718], [0.26652, 0.32768, 0.76412], [0.26816, 0.33825, 0.78050],
  [0.26967, 0.34878, 0.79631], [0.27103, 0.35926, 0.81156], [0.27226, 0.36970, 0.82624], [0.27334, 0.38008, 0.84037], [0.27429, 0.39043, 0.85393],
  [0.27509, 0.40072, 0.86692], [0.27576, 0.41097, 0.87936], [0.27628, 0.42118, 0.89123], [0.27667, 0.43134, 0.90254], [0.27691, 0.44145, 0.91328],
  [0.27701, 0.45152, 0.92347], [0.27698, 0.46153, 0.93309], [0.27680, 0.47151, 0.94214], [0.27648, 0.48144, 0.95064], [0.27603, 0.49132, 0.95857],
  [0.27543, 0.50115, 0.96594], [0.27469, 0.51094, 0.97275], [0.27381, 0.52069, 0.97899], [0.27273, 0.53040, 0.98461], [0.27106, 0.54015, 0.98930],
  [0.26878, 0.54995, 0.99303], [0.26592, 0.55979, 0.99583], [0.26252, 0.56967, 0.99773], [0.25862, 0.57958, 0.99876], [0.25425, 0.58950, 0.99896],
  [0.24946, 0.59943, 0.99835], [0.24427, 0.60937, 0.99697], [0.23874, 0.61931, 0.99485], [0.23288, 0.62923, 0.99202], [0.22676, 0.63913, 0.98851],
  [0.22039, 0.64901, 0.98436], [0.21382, 0.65886, 0.97959], [0.20708, 0.66866, 0.97423], [0.20021, 0.67842, 0.96833], [0.19326, 0.68812, 0.96190],
  [0.18625, 0.69775, 0.95498], [0.17923, 0.70732, 0.94761], [0.17223, 0.71680, 0.93981], [0.16529, 0.72620, 0.93161], [0.15844, 0.73551, 0.92305],
  [0.15173, 0.74472, 0.91416], [0.14519, 0.75381, 0.90496], [0.13886, 0.76279, 0.89550], [0.13278, 0.77165, 0.88580], [0.12698, 0.78037, 0.87590],
  [0.12151, 0.78896, 0.86581], [0.11639, 0.79740, 0.85559], [0.11167, 0.80569, 0.84525], [0.10738, 0.81381, 0.83484], [0.10357, 0.82177, 0.82437],
  [0.10026, 0.82955, 0.81389], [0.09750, 0.83714, 0.80342], [0.09532, 0.84455, 0.79299], [0.09377, 0.85175, 0.78264], [0.09287, 0.85875, 0.77240],
  [0.09267, 0.86554, 0.76230], [0.09320, 0.87211, 0.75237], [0.09451, 0.87844, 0.74265], [0.09662, 0.88454, 0.73316], [0.09958, 0.89040, 0.72393],
  [0.10342, 0.89600, 0.71500], [0.10815, 0.90142, 0.70599], [0.11374, 0.90673, 0.69651], [0.12014, 0.91193, 0.68660], [0.12733, 0.91701, 0.67627],
  [0.13526, 0.92197, 0.66556], [0.14391, 0.92680, 0.65448], [0.15323, 0.93151, 0.64308], [0.16319, 0.93609, 0.63137], [0.17377, 0.94053, 0.61938],
  [0.18491, 0.94484, 0.60713], [0.19659, 0.94901, 0.59466], [0.20877, 0.95304, 0.58199], [0.22142, 0.95692, 0.56914], [0.23449, 0.96065, 0.55614],
  [0.24797, 0.96423, 0.54303], [0.26180, 0.96765, 0.52981], [0.27597, 0.97092, 0.51653], [0.29042, 0.97403, 0.50321], [0.30513, 0.97697, 0.48987],
  [0.32006, 0.97974, 0.47654], [0.33517, 0.98234, 0.46325], [0.35043, 0.98477, 0.45002], [0.36581, 0.98702, 0.43688], [0.38127, 0.98909, 0.42386],
  [0.39678, 0.99098, 0.41098], [0.41229, 0.99268, 0.39826], [0.42778, 0.99419, 0.38575], [0.44321, 0.99551, 0.37345], [0.45854, 0.99663, 0.36140],
  [0.47375, 0.99755, 0.34963], [0.48879, 0.99828, 0.33816], [0.50362, 0.99879, 0.32701], [0.51822, 0.99910, 0.31622], [0.53255, 0.99919, 0.30581],
  [0.54658, 0.99907, 0.29581], [0.56026, 0.99873, 0.28623], [0.57357, 0.99817, 0.27712], [0.58646, 0.99739, 0.26849], [0.59891, 0.99638, 0.26038],
  [0.61088, 0.99514, 0.25280], [0.62233, 0.99366, 0.24579], [0.63323, 0.99195, 0.23937], [0.64362, 0.98999, 0.23356], [0.65394, 0.98775, 0.22835],
  [0.66428, 0.98524, 0.22370], [0.67462, 0.98246, 0.21960], [0.68494, 0.97941, 0.21602], [0.69525, 0.97610, 0.21294], [0.70553, 0.97255, 0.21032],
  [0.71577, 0.96875, 0.20815], [0.72596, 0.96470, 0.20640], [0.73610, 0.96043, 0.20504], [0.74617, 0.95593, 0.20406], [0.75617, 0.95121, 0.20343],
  [0.76608, 0.94627, 0.20311], [0.77591, 0.94113, 0.20310], [0.78563, 0.93579, 0.20336], [0.79524, 0.93025, 0.20386], [0.80473, 0.92452, 0.20459],
  [0.81410, 0.91861, 0.20552], [0.82333, 0.91253, 0.20663], [0.83241, 0.90627, 0.20788], [0.84133, 0.89986, 0.20926], [0.85010, 0.89328, 0.21074],
  [0.85868, 0.88655, 0.21230], [0.86709, 0.87968, 0.21391], [0.87530, 0.87267, 0.21555], [0.88331, 0.86553, 0.21719], [0.89112, 0.85826, 0.21880],
  [0.89870, 0.85087, 0.22038], [0.90605, 0.84337, 0.22188], [0.91317, 0.83576, 0.22328], [0.92004, 0.82806, 0.22456], [0.92666, 0.82025, 0.22570],
  [0.93301, 0.81236, 0.22667], [0.93909, 0.80439, 0.22744], [0.94489, 0.79634, 0.22800], [0.95039, 0.78823, 0.22831], [0.95560, 0.78005, 0.22836],
  [0.96049, 0.77181, 0.22811], [0.96507, 0.76352, 0.22754], [0.96931, 0.75519, 0.22663], [0.97323, 0.74682, 0.22536], [0.97679, 0.73842, 0.22369],
  [0.98000, 0.73000, 0.22161], [0.98289, 0.72140, 0.21918], [0.98549, 0.71250, 0.21650], [0.98781, 0.70330, 0.21358], [0.98986, 0.69382, 0.21043],
  [0.99163, 0.68408, 0.20706], [0.99314, 0.67408, 0.20348], [0.99438, 0.66386, 0.19971], [0.99535, 0.65341, 0.19577], [0.99606, 0.64277, 0.19165],
  [0.99654, 0.63193, 0.18738], [0.99675, 0.62093, 0.18297], [0.99672, 0.60977, 0.17842], [0.99644, 0.59846, 0.17376], [0.99593, 0.58703, 0.16899],
  [0.99517, 0.57549, 0.16412], [0.99419, 0.56386, 0.15918], [0.99297, 0.55214, 0.15417], [0.99153, 0.54036, 0.14910], [0.98987, 0.52854, 0.14398],
  [0.98799, 0.51667, 0.13883], [0.98590, 0.50479, 0.13367], [0.98360, 0.49291, 0.12849], [0.98108, 0.48104, 0.12332], [0.97837, 0.46920, 0.11817],
  [0.97545, 0.45740, 0.11305], [0.97234, 0.44565, 0.10797], [0.96904, 0.43399, 0.10294], [0.96555, 0.42241, 0.09798], [0.96187, 0.41093, 0.09310],
  [0.95801, 0.39958, 0.08831], [0.95398, 0.38836, 0.08362], [0.94977, 0.37729, 0.07905], [0.94538, 0.36638, 0.07461], [0.94084, 0.35566, 0.07031],
  [0.93612, 0.34513, 0.06616], [0.93125, 0.33482, 0.06218], [0.92623, 0.32473, 0.05837], [0.92105, 0.31489, 0.05475], [0.91572, 0.30530, 0.05134],
  [0.91024, 0.29599, 0.04814], [0.90463, 0.28696, 0.04516], [0.89888, 0.27824, 0.04243], [0.89298, 0.26981, 0.03993], [0.88691, 0.26152, 0.03753],
  [0.88066, 0.25334, 0.03521], [0.87422, 0.24526, 0.03297], [0.86760, 0.23730, 0.03082], [0.86079, 0.22945, 0.02875], [0.85380, 0.22170, 0.02677],
  [0.84662, 0.21407, 0.02487], [0.83926, 0.20654, 0.02305], [0.83172, 0.19912, 0.02131], [0.82399, 0.19182, 0.01966], [0.81608, 0.18462, 0.01809],
  [0.80799, 0.17753, 0.01660], [0.79971, 0.17056, 0.01520], [0.79125, 0.16368, 0.01387], [0.78260, 0.15693, 0.01264], [0.77377, 0.15028, 0.01148],
  [0.76476, 0.14374, 0.01041], [0.75556, 0.13731, 0.00942], [0.74617, 0.13098, 0.00851], [0.73661, 0.12477, 0.00769], [0.72686, 0.11867, 0.00695],
  [0.71692, 0.11268, 0.00629], [0.70680, 0.10680, 0.00571], [0.69650, 0.10102, 0.00522], [0.68602, 0.09536, 0.00481], [0.67535, 0.08980, 0.00449],
  [0.66449, 0.08436, 0.00424], [0.65345, 0.07902, 0.00408], [0.64223, 0.07380, 0.00401], [0.63082, 0.06868, 0.00401], [0.61923, 0.06367, 0.00410],
  [0.60746, 0.05878, 0.00427], [0.59550, 0.05399, 0.00453], [0.58336, 0.04931, 0.00486], [0.57103, 0.04474, 0.00529], [0.55852, 0.04028, 0.00579],
  [0.54583, 0.03593, 0.00638], [0.53295, 0.03169, 0.00705], [0.51989, 0.02756, 0.00780], [0.50664, 0.02354, 0.00863], [0.49321, 0.01963, 0.00955],
  [0.47960, 0.01583, 0.01055]
];

/* グレースケール値（0-1）をturboカラーマップに変換 */
function grayToTurbo(grayValue) {
  // 0-1の範囲にクランプ
  const value = Math.max(0, Math.min(1, grayValue));
  
  // カラーマップのインデックスを計算
  const index = Math.floor(value * (TURBO_COLORMAP.length - 1));
  const nextIndex = Math.min(index + 1, TURBO_COLORMAP.length - 1);
  
  // 線形補間のための係数
  const t = value * (TURBO_COLORMAP.length - 1) - index;
  
  // 線形補間でRGB値を計算
  const color1 = TURBO_COLORMAP[index];
  const color2 = TURBO_COLORMAP[nextIndex];
  
  const r = color1[0] + t * (color2[0] - color1[0]);
  const g = color1[1] + t * (color2[1] - color1[1]);
  const b = color1[2] + t * (color2[2] - color1[2]);
  
  return [
    Math.round(r * 255),
    Math.round(g * 255),
    Math.round(b * 255)
  ];
}

/* モード切替の処理 */
function switchMode(mode) {
  currentMode = mode;
  
  if (mode === 'slider') {
    sliderContainer.style.display = 'block';
    interactiveContainer.style.display = 'none';
    targetSection.style.display = 'block';
    
    // スライダーモード用のイベントリスナーを有効化
    imageSlider.addEventListener('input', onSliderChange);
    
    // インタラクティブモード用のイベントリスナーを無効化
    inputCanvas.removeEventListener('mousemove', onMouseMove);
    inputCanvas.removeEventListener('mousedown', onMouseDown);
    inputCanvas.removeEventListener('mouseup', onMouseUp);
    inputCanvas.removeEventListener('click', onMouseClick);
    inputCanvas.style.cursor = 'default';
    
    // キャンバスをクリアしてスライダーモードの現在の画像を再描画
    inCtx.clearRect(0, 0, SIZE, SIZE);
    outCtx.clearRect(0, 0, SIZE, SIZE);
    targetCtx.clearRect(0, 0, SIZE, SIZE);
  } else if (mode === 'interactive') {
    sliderContainer.style.display = 'none';
    interactiveContainer.style.display = 'block';
    targetSection.style.display = 'none';
    
    // スライダーモード用のイベントリスナーを無効化
    imageSlider.removeEventListener('input', onSliderChange);
    
    // インタラクティブモード用のイベントリスナーを有効化
    inputCanvas.addEventListener('mousemove', onMouseMove);
    inputCanvas.addEventListener('mousedown', onMouseDown);
    inputCanvas.addEventListener('mouseup', onMouseUp);
    inputCanvas.addEventListener('click', onMouseClick);
    inputCanvas.style.cursor = 'crosshair';
    
    // キャンバスをクリア
    inCtx.clearRect(0, 0, SIZE, SIZE);
    outCtx.clearRect(0, 0, SIZE, SIZE);
    targetCtx.clearRect(0, 0, SIZE, SIZE);
  }
}

/* インタラクティブモードの初期化 */
async function initInteractiveMode() {
  // 入力データを初期化（全て0）
  interactiveInputData.fill(0);
  permanentInputData.fill(0);
  isMouseDown = false;
  
  // キャンバスをクリア
  inCtx.clearRect(0, 0, SIZE, SIZE);
  outCtx.clearRect(0, 0, SIZE, SIZE);
  
  // マスクが読み込まれていない場合は読み込む
  if (!pressureMask || !sedMask) {
    console.log('Loading masks for interactive mode...');
    await loadMasks();
  }
  
  // 初期状態のキャンバスを更新（空の状態を視覚化）
  updateInteractiveCanvas();
  
  // 初期推論を実行
  performInteractiveInference();
}

/* ガウス分布を生成する関数 */
function generateGaussian(centerX, centerY, amplitude = 0.5, sigma = 8.0, addToPermanent = false) {
  const tempData = new Float32Array(SIZE * SIZE);
  
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const dx = x - centerX;
      const dy = y - centerY;
      const distance2 = dx * dx + dy * dy;
      const gaussValue = amplitude * Math.exp(-distance2 / (2 * sigma * sigma));
      
      const index = y * SIZE + x;
      tempData[index] = gaussValue;
      
      // マスクされた部分は常にゼロに設定
      if (pressureMask) {
        const p = index * 4;
        const maskValue = pressureMask.data[p];
        // しきい値128で黒い部分がマスクされる部分（表示されない部分）
        if (maskValue >= 128) {
          tempData[index] = 0;
        }
      }
    }
  }
  
  // 永続的なデータに追加するかどうか
  if (addToPermanent) {
    for (let i = 0; i < SIZE * SIZE; i++) {
      permanentInputData[i] = Math.min(1.0, permanentInputData[i] + tempData[i]);
    }
  }
  
  // 最終的な入力データを作成（永続的なデータ + 一時的なデータ）
  for (let i = 0; i < SIZE * SIZE; i++) {
    interactiveInputData[i] = Math.min(1.0, permanentInputData[i] + tempData[i]);
  }
}

/* インタラクティブモード用のマウス移動処理 */
function onMouseMove(event) {
  if (currentMode !== 'interactive' || isProcessing) return;
  
  const rect = inputCanvas.getBoundingClientRect();
  const x = Math.floor((event.clientX - rect.left) * (SIZE / rect.width));
  const y = Math.floor((event.clientY - rect.top) * (SIZE / rect.height));
  
  if (isMouseDown) {
    // マウスボタンが押されている場合：永続的なデータに追加（より強い分布）
    generateGaussian(x, y, 0.3, 8.0, true);
  } else {
    // マウスボタンが離されている場合：一時的な表示のみ（より弱い分布）
    // まず永続的なデータのみで interactiveInputData をリセット
    for (let i = 0; i < SIZE * SIZE; i++) {
      interactiveInputData[i] = permanentInputData[i];
    }
    // 現在のマウス位置に一時的なガウス分布を追加
    generateGaussian(x, y, 0.3, 8.0, false);
  }
  
  // キャンバスを更新
  updateInteractiveCanvas();
  
  // 推論を実行
  performInteractiveInference();
}

/* インタラクティブモード用のマウスダウン処理 */
function onMouseDown(event) {
  if (currentMode !== 'interactive' || isProcessing) return;
  
  isMouseDown = true;
}

/* インタラクティブモード用のマウスアップ処理 */
function onMouseUp(event) {
  if (currentMode !== 'interactive') return;
  
  isMouseDown = false;
}

/* インタラクティブモード用のマウスクリック処理 */
function onMouseClick(event) {
  if (currentMode !== 'interactive' || isProcessing) return;
  
  const rect = inputCanvas.getBoundingClientRect();
  const x = Math.floor((event.clientX - rect.left) * (SIZE / rect.width));
  const y = Math.floor((event.clientY - rect.top) * (SIZE / rect.height));
  
  // より強いガウス分布を永続的に追加
  generateGaussian(x, y, 0.3, 8.0, true);
  
  // キャンバスを更新
  updateInteractiveCanvas();
  
  // 推論を実行
  performInteractiveInference();
}

/* インタラクティブモード用のキャンバス更新 */
function updateInteractiveCanvas() {
  // マスクされた部分を確実にゼロに設定
  if (pressureMask) {
    for (let i = 0; i < SIZE * SIZE; i++) {
      const p = i * 4;
      const maskValue = pressureMask.data[p];
      // しきい値128で黒い部分がマスクされる部分（表示されない部分）
      if (maskValue >= 128) {
        interactiveInputData[i] = 0;
      }
    }
  }
  
  // 入力データをturboカラーマップで表示
  const inputTurboImageData = new ImageData(SIZE, SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const grayValue = interactiveInputData[i]; // 0-1の正規化済みグレースケール値
    const [r, g, b] = grayToTurbo(grayValue);
    const p = i * 4;
    inputTurboImageData.data[p]     = r;   // R
    inputTurboImageData.data[p + 1] = g;   // G
    inputTurboImageData.data[p + 2] = b;   // B
    inputTurboImageData.data[p + 3] = 255; // A (完全不透明)
  }
  
  // マスクを適用
  const maskedInputImageData = applyMask(inputTurboImageData, pressureMask);
  inCtx.putImageData(maskedInputImageData, 0, 0);
}

/* インタラクティブモード用の推論実行 */
async function performInteractiveInference() {
  if (isProcessing) return;
  
  try {
    isProcessing = true;
    
    // マスクが読み込まれていない場合は読み込む
    if (!pressureMask || !sedMask) {
      console.log('Loading masks...');
      await loadMasks();
    }

    console.log('Loading ONNX session...');
    const session = await sessionPromise;
    
    // 入力テンソルを作成
    const input = new ort.Tensor('float32', interactiveInputData, [1, 1, SIZE, SIZE]);

    /* 推論（時間計測あり） */
    console.log('Starting interactive inference...');
    const startTime = performance.now();
    const { output } = await session.run({ input });
    const endTime = performance.now();
    const inferenceTimeMs = endTime - startTime;
    
    console.log(`Interactive inference completed in ${inferenceTimeMs.toFixed(2)} ms`);
    
    /* 推論時間を表示に更新 */
    inferenceTime.textContent = `Inference time: ${inferenceTimeMs.toFixed(2)} ms`;

    /* モデル出力をturboカラーマップで描画 */
    const outputImageData = new ImageData(SIZE, SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      const grayValue = Math.max(0, Math.min(1, output.data[i])); // 0-1の範囲にクランプ
      const [r, g, b] = grayToTurbo(grayValue);
      const p = i * 4;
      outputImageData.data[p]     = r;   // R
      outputImageData.data[p + 1] = g;   // G
      outputImageData.data[p + 2] = b;   // B
      outputImageData.data[p + 3] = 255; // A (完全不透明)
    }

    /* 表示用：出力画像にSEDマスクを適用 */
    const maskedOutputImageData = applyMask(outputImageData, sedMask);
    outCtx.putImageData(maskedOutputImageData, 0, 0);
  } catch (error) {
    console.error('Error during interactive inference:', error);
  } finally {
    isProcessing = false;
  }
}

/* モード切替のイベントリスナー */
sliderMode.addEventListener('change', async () => {
  if (sliderMode.checked) {
    switchMode('slider');
    // スライダーモードに切り替えた時は現在選択されている画像を再描画
    if (!isProcessing && preloadedImages.length > 0) {
      const currentIndex = parseInt(imageSlider.value);
      const img = preloadedImages[currentIndex];
      const targetImg = preloadedTargetImages[currentIndex];
      if (img) {
        updateImageInfo(currentIndex);
        await processImage(img, targetImg);
      }
    }
  }
});

interactiveMode.addEventListener('change', async () => {
  if (interactiveMode.checked) {
    switchMode('interactive');
    // ユーザーが手動で切り替えた時のみインタラクティブモードを初期化
    if (!isProcessing) {
      await initInteractiveMode();
    }
  }
});

/* クリアボタンの処理 */
clearButton.addEventListener('click', async () => {
  if (currentMode === 'interactive') {
    await initInteractiveMode();
  }
});

/* ページ読み込み時にマスク画像と入力画像をプリロードし、デフォルト画像を処理 */
Promise.all([loadMasks(), preloadInputImages(), preloadTargetImages()])
  .then(async () => {
    console.log('All resources loaded, now initializing mode...');
    
    // ラジオボタンの状態を確認して適切なモードを設定
    if (interactiveMode.checked) {
      switchMode('interactive');
      await initInteractiveMode();
    } else {
      switchMode('slider');
      await loadDefaultImage();
    }
  })
  .catch(error => {
    console.error('Failed to initialize application:', error);
    console.log('Some features may not work properly.');
  });
