const { app, BrowserWindow, shell } = require('electron');
const path = require('path');
const http = require('http');
const { spawn, execSync } = require('child_process');
const fs = require('fs');

let mainWindow = null;
let splashWindow = null;
let backendProcess = null;
const SERVER_URL = 'http://localhost:8000';
const HEALTH_URL = 'http://localhost:8000/api/health';

function isServerHealthy() {
  return new Promise((resolve) => {
    const req = http.get(HEALTH_URL, (res) => {
      resolve(res.statusCode === 200);
    });
    req.on('error', () => resolve(false));
    req.setTimeout(800, () => {
      req.destroy();
      resolve(false);
    });
  });
}

function resolveBackendExecutable() {
  // 1. Packaged Electron app extraResources path
  if (app.isPackaged) {
    const packagedExe = path.join(process.resourcesPath, 'backend', 'AIShortsGenerator.exe');
    if (fs.existsSync(packagedExe)) {
      return { type: 'exe', path: packagedExe, cwd: path.dirname(packagedExe) };
    }
  }

  // 2. In development mode, prioritize live server.py
  const serverPy = path.join(__dirname, '..', 'server.py');
  if (fs.existsSync(serverPy)) {
    return { type: 'python', path: serverPy, cwd: path.join(__dirname, '..') };
  }

  // 3. Fallback: local dist folder from PyInstaller build
  const distExe = path.join(__dirname, '..', 'dist', 'AIShortsGenerator', 'AIShortsGenerator.exe');
  if (fs.existsSync(distExe)) {
    return { type: 'exe', path: distExe, cwd: path.dirname(distExe) };
  }

  return null;
}

function startBackendServer() {
  const backend = resolveBackendExecutable();
  if (!backend) {
    console.warn('[Electron] No backend executable or script found.');
    return;
  }

  console.log(`[Electron] Launching backend: ${backend.path} (${backend.type})`);

  const spawnEnv = { ...process.env, ELECTRON_RUN: '1' };

  if (backend.type === 'exe') {
    backendProcess = spawn(backend.path, ['--no-browser'], {
      cwd: backend.cwd,
      windowsHide: true,
      env: spawnEnv,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  } else {
    // Run via Python
    const pyCmd = process.platform === 'win32' ? 'python' : 'python3';
    backendProcess = spawn(pyCmd, [backend.path, '--no-browser'], {
      cwd: backend.cwd,
      windowsHide: true,
      env: spawnEnv,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  }

  backendProcess.stdout.on('data', (data) => {
    const msg = data.toString().trim();
    if (msg) console.log(`[Backend] ${msg}`);
  });

  backendProcess.stderr.on('data', (data) => {
    const msg = data.toString().trim();
    if (msg) console.error(`[Backend Err] ${msg}`);
  });

  backendProcess.on('exit', (code, signal) => {
    console.log(`[Electron] Backend process exited with code=${code}, signal=${signal}`);
    backendProcess = null;
  });
}

function stopBackendServer() {
  if (backendProcess && backendProcess.pid) {
    console.log(`[Electron] Terminating backend process tree (PID: ${backendProcess.pid})...`);
    try {
      if (process.platform === 'win32') {
        execSync(`taskkill /pid ${backendProcess.pid} /T /F`, { stdio: 'ignore' });
      } else {
        backendProcess.kill('SIGTERM');
      }
    } catch (err) {
      // Process may already be dead
    }
    backendProcess = null;
  }
}

function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 380,
    height: 300,
    frame: false,
    resizable: false,
    center: true,
    alwaysOnTop: true,
    show: false,
    backgroundColor: '#09090b',
    icon: path.join(__dirname, '..', 'web', 'assets', 'icon.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  splashWindow.loadFile(path.join(__dirname, 'splash.html'));
  splashWindow.once('ready-to-show', () => {
    if (splashWindow) splashWindow.show();
  });
}

function createMainWindow() {
  const iconPath = process.platform === 'win32'
    ? path.join(__dirname, '..', 'web', 'assets', 'icon.ico')
    : path.join(__dirname, '..', 'web', 'assets', 'icon.png');

  mainWindow = new BrowserWindow({
    width: 1360,
    height: 860,
    minWidth: 1060,
    minHeight: 700,
    show: false,
    backgroundColor: '#09090b',
    title: 'A-S-G Studio - AI Shorts Generator',
    icon: iconPath,
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true,
    },
  });

  // Open external links in default browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('http:') || url.startsWith('https:')) {
      shell.openExternal(url);
    }
    return { action: 'deny' };
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
    app.quit();
  });
}

async function waitForServerAndLaunch() {
  createSplashWindow();
  createMainWindow();

  // Check if server is already running
  let healthy = await isServerHealthy();
  if (!healthy) {
    startBackendServer();
  }

  // Poll until server is healthy (timeout after 45s)
  const startTime = Date.now();
  const maxWaitMs = 45000;

  while (!healthy && Date.now() - startTime < maxWaitMs) {
    await new Promise((r) => setTimeout(r, 600));
    healthy = await isServerHealthy();
  }

  if (!healthy) {
    console.error('[Electron] Failed to connect to backend server within timeout.');
  }

  // Load backend web UI
  mainWindow.loadURL(SERVER_URL);

  mainWindow.webContents.once('did-finish-load', () => {
    setTimeout(() => {
      if (splashWindow && !splashWindow.isDestroyed()) {
        splashWindow.destroy();
        splashWindow = null;
      }
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.show();
        mainWindow.focus();
      }
    }, 400);
  });
}

const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(waitForServerAndLaunch);

  app.on('window-all-closed', () => {
    stopBackendServer();
    if (process.platform !== 'darwin') {
      app.quit();
    }
  });

  app.on('will-quit', () => {
    stopBackendServer();
  });
}
