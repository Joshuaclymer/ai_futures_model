/**
 * Caching system for AI futures simulation runs.
 * Uses file-based caching to persist results across server restarts.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

const CACHE_DIR = path.join(process.cwd(), '.simulation-cache');
const CACHE_TTL_MS = 24 * 60 * 60 * 1000; // 24 hours

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  paramHash: string;
}

// Ensure cache directory exists
function ensureCacheDir(): void {
  if (!fs.existsSync(CACHE_DIR)) {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
  }
}

// Generate a hash from parameters for cache key
export function generateParamHash(params: Record<string, unknown>): string {
  // Sort keys to ensure consistent hashing
  const sortedParams = Object.keys(params)
    .sort()
    .reduce((acc, key) => {
      acc[key] = params[key];
      return acc;
    }, {} as Record<string, unknown>);

  const paramString = JSON.stringify(sortedParams);
  return crypto.createHash('md5').update(paramString).digest('hex');
}

// Get cache file path for a given hash
function getCacheFilePath(hash: string): string {
  return path.join(CACHE_DIR, `sim_${hash}.json`);
}

// Get cached simulation result if it exists and is valid
export function getCachedSimulation<T>(params: Record<string, unknown>): T | null {
  try {
    ensureCacheDir();
    const hash = generateParamHash(params);
    const filePath = getCacheFilePath(hash);

    if (!fs.existsSync(filePath)) {
      return null;
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    const entry: CacheEntry<T> = JSON.parse(content);

    // Check if cache is still valid
    const age = Date.now() - entry.timestamp;
    if (age > CACHE_TTL_MS) {
      // Cache expired, delete it
      fs.unlinkSync(filePath);
      return null;
    }

    console.log(`[Cache] Hit for hash ${hash.substring(0, 8)}... (age: ${Math.round(age / 1000 / 60)}min)`);
    return entry.data;
  } catch (error) {
    console.error('[Cache] Error reading cache:', error);
    return null;
  }
}

// Save simulation result to cache
export function cacheSimulation<T>(params: Record<string, unknown>, data: T): void {
  try {
    ensureCacheDir();
    const hash = generateParamHash(params);
    const filePath = getCacheFilePath(hash);

    const entry: CacheEntry<T> = {
      data,
      timestamp: Date.now(),
      paramHash: hash,
    };

    fs.writeFileSync(filePath, JSON.stringify(entry), 'utf-8');
    console.log(`[Cache] Saved for hash ${hash.substring(0, 8)}...`);
  } catch (error) {
    console.error('[Cache] Error writing cache:', error);
  }
}

// Clear all cached simulations
export function clearSimulationCache(): number {
  try {
    ensureCacheDir();
    const files = fs.readdirSync(CACHE_DIR).filter(f => f.startsWith('sim_') && f.endsWith('.json'));
    files.forEach(file => {
      fs.unlinkSync(path.join(CACHE_DIR, file));
    });
    console.log(`[Cache] Cleared ${files.length} cached simulations`);
    return files.length;
  } catch (error) {
    console.error('[Cache] Error clearing cache:', error);
    return 0;
  }
}

// Get cache statistics
export function getCacheStats(): { count: number; totalSize: number; oldestAge: number } {
  try {
    ensureCacheDir();
    const files = fs.readdirSync(CACHE_DIR).filter(f => f.startsWith('sim_') && f.endsWith('.json'));

    let totalSize = 0;
    let oldestTimestamp = Date.now();

    files.forEach(file => {
      const filePath = path.join(CACHE_DIR, file);
      const stats = fs.statSync(filePath);
      totalSize += stats.size;

      try {
        const content = fs.readFileSync(filePath, 'utf-8');
        const entry = JSON.parse(content);
        if (entry.timestamp < oldestTimestamp) {
          oldestTimestamp = entry.timestamp;
        }
      } catch {
        // Ignore parsing errors
      }
    });

    return {
      count: files.length,
      totalSize,
      oldestAge: Date.now() - oldestTimestamp,
    };
  } catch (error) {
    console.error('[Cache] Error getting stats:', error);
    return { count: 0, totalSize: 0, oldestAge: 0 };
  }
}
