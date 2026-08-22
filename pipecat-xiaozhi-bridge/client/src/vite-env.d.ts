/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_WS_PATH?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
