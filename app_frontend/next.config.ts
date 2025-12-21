import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  cacheComponents: true,
  rewrites: async () => {
    // Only proxy specific backend API routes, not local Next.js API routes
    const backendRoutes = [
      'run-sw-progress-simulation',
      'sampling-config',
    ];

    if (process.env.NODE_ENV !== 'development') {
      return [];
    }

    return backendRoutes.map(route => ({
      source: `/api/${route}`,
      destination: `http://127.0.0.1:5329/api/${route}`,
    }));
  },
  turbopack: {
    root: __dirname,
    rules: {
      '**/*.svg': {
        "loaders": ["@svgr/webpack"],
        "as": "*.js"
      },
      '**/*.html': {
        "loaders": ["html-loader"],
        "as": "*.js"
      }
    }
  },
  webpack: (config) => {
    config.module.rules.push({
      test: /\.svg$/,
      use: ["@svgr/webpack"],
    });
    config.module.rules.push({
      test: /\.html$/,
      use: "html-loader",
    });
    return config;
  },
    reactCompiler: true,
};

export default nextConfig;