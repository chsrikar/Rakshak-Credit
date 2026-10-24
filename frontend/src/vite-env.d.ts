/// <reference types="vite/client" />

declare module '*.css' {
  const content: Record<string, string>;
  export default content;
}

declare module '*.jsx' {
  const component: any;
  export default component;
}

declare module './components/BorrowerUnderwriting' {
  const BorrowerUnderwriting: any;
  export default BorrowerUnderwriting;
}
