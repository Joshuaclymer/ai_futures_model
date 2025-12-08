import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execFile } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function getPythonExecutable() {
  return process.env.PYTHON || process.env.PYTHON_EXECUTABLE || 'python3';
}

async function main() {
  const pythonScript = path.join(__dirname, 'python_dump_parameter_config.py');
  if (!fs.existsSync(pythonScript)) {
    throw new Error(`Python parameter dump script not found at ${pythonScript}`);
  }

  const { stdout, stderr } = await execFileAsync(getPythonExecutable(), [pythonScript], {
    maxBuffer: 1024 * 1024 * 10
  });

  if (stderr.trim().length > 0) {
    console.warn('⚠️ Python config generator emitted warnings:', stderr.trim());
  }

  let payload;
  try {
    payload = JSON.parse(stdout);
  } catch (error) {
    console.error('Failed to parse Python parameter config payload');
    throw error;
  }

  const outputDir = path.join(__dirname, '..', 'config');
  fs.mkdirSync(outputDir, { recursive: true });
  const outputPath = path.join(outputDir, 'python-parameter-config.json');
  fs.writeFileSync(outputPath, JSON.stringify(payload, null, 2));
  console.log(`✅ Wrote Python parameter config to ${outputPath}`);
}

main().catch(error => {
  console.error('❌ Failed to generate Python-aligned parameter config:', error);
  process.exit(1);
});
