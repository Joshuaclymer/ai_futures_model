import Link from 'next/link';
import fs from 'fs';
import path from 'path';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import yaml from 'js-yaml';

export const metadata = {
  title: 'Model Parameters',
  description: 'What the parameters represent and how we chose their values.',
};

// Category configuration mapping directory names to display info
const categoryConfig: Record<string, { id: string; title: string; order: number }> = {
  'perceptions_parameters': { id: 'detection', title: 'Detection Parameters', order: 1 },
  'compute_parameters': { id: 'compute', title: 'Compute and Chips', order: 2 },
  'data_center_and_energy_parameters': { id: 'datacenters', title: 'Datacenters and Energy', order: 3 },
  'black_project_parameters': { id: 'black-project', title: 'Black Project Configuration', order: 4 },
};

// Load monte carlo parameters and extract CI values
function loadMonteCarloCI(): Record<string, string> {
  const ciMap: Record<string, string> = {};

  try {
    const yamlPath = path.join(process.cwd(), '..', 'ai_futures_simulator', 'parameters', 'monte_carlo_parameters.yaml');
    const yamlContent = fs.readFileSync(yamlPath, 'utf-8');
    const config = yaml.load(yamlContent) as Record<string, unknown>;

    // Recursively extract ci80 values from the yaml structure
    function extractCI(obj: unknown, prefix: string = '') {
      if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
        const record = obj as Record<string, unknown>;
        for (const [key, value] of Object.entries(record)) {
          if (value && typeof value === 'object' && !Array.isArray(value)) {
            const valueRecord = value as Record<string, unknown>;
            if ('ci80' in valueRecord && Array.isArray(valueRecord.ci80)) {
              const ci = valueRecord.ci80 as number[];
              ciMap[key] = `[${ci[0]}, ${ci[1]}]`;
            } else {
              extractCI(value, key);
            }
          }
        }
      }
    }

    extractCI(config);
  } catch {
    // Yaml file doesn't exist or can't be parsed
  }

  return ciMap;
}

// Read markdown files from the parameter_documentation folder by scanning subdirectories
function getParameterDocs(): Record<string, { title: string; docs: { filename: string; paramName: string; content: string }[] }> {
  const docsPath = path.join(process.cwd(), '..', 'ai_futures_simulator', 'parameters', 'parameter_documentation');
  const ciMap = loadMonteCarloCI();

  const categories: Record<string, { title: string; docs: { filename: string; content: string }[]; order: number }> = {};

  // Scan each category directory
  try {
    const categoryDirs = fs.readdirSync(docsPath, { withFileTypes: true })
      .filter(dirent => dirent.isDirectory() && categoryConfig[dirent.name]);

    for (const categoryDir of categoryDirs) {
      const config = categoryConfig[categoryDir.name];
      const categoryPath = path.join(docsPath, categoryDir.name);
      const docs: { filename: string; content: string }[] = [];

      // Scan each parameter subdirectory within the category
      const paramDirs = fs.readdirSync(categoryPath, { withFileTypes: true })
        .filter(dirent => dirent.isDirectory());

      for (const paramDir of paramDirs) {
        const paramPath = path.join(categoryPath, paramDir.name);
        // Look for markdown file with same name as directory
        const mdFilename = `${paramDir.name}.md`;
        const mdPath = path.join(paramPath, mdFilename);

        try {
          if (fs.existsSync(mdPath)) {
            let content = fs.readFileSync(mdPath, 'utf-8');
            // Strip the first H1 heading from markdown (we use paramName as title instead)
            content = content.replace(/^#\s+.+\n+/, '');

            // Replace "Range" header with "CI" in tables
            content = content.replace(/\|\s*Range\s*\|/gi, '| CI |');
            content = content.replace(/\|\s*-+\s*\|(\s*-+\s*\|)(\s*-+\s*\|)(\s*-+\s*\|)/g, (match) => match);

            // Get CI value for this parameter from yaml, or use N/A
            const paramName = paramDir.name;
            const ciValue = ciMap[paramName] || 'N/A';

            // Replace the range column values in table rows with CI from yaml
            // Match table rows: | text | text | range_value | text |
            content = content.replace(
              /\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|/g,
              (match, col1, col2, col3, col4) => {
                // Skip header row (contains "Range" or "CI") and separator row (contains only dashes)
                if (col3.includes('Range') || col3.includes('CI') || /^[\s-]+$/.test(col3)) {
                  return match;
                }
                return `|${col1}|${col2}| ${ciValue} |${col4}|`;
              }
            );

            docs.push({ filename: mdFilename, paramName, content });
          }
        } catch {
          // File doesn't exist or can't be read, skip it
        }
      }

      if (docs.length > 0) {
        categories[config.id] = {
          title: config.title,
          docs,
          order: config.order,
        };
      }
    }
  } catch {
    // Directory doesn't exist or can't be read
  }

  // Sort categories by order and return without the order field
  const sortedEntries = Object.entries(categories)
    .sort(([, a], [, b]) => a.order - b.order);

  const result: Record<string, { title: string; docs: { filename: string; paramName: string; content: string }[] }> = {};
  for (const [id, { title, docs }] of sortedEntries) {
    result[id] = { title, docs };
  }

  return result;
}

// Category icons for visual interest
const categoryIcons: Record<string, string> = {
  'detection': '◉',
  'compute': '⬡',
  'datacenters': '⚡',
  'black-project': '◇',
};

export default function BlackProjectParametersPage() {
  const categories = getParameterDocs();
  const categoryEntries = Object.entries(categories);

  return (
    <div className="min-h-screen bg-[#fffff8]">
      {/* Elegant Header */}
      <header className="sticky top-0 z-50 bg-[#fffff8]/95 backdrop-blur-sm border-b border-stone-200/60">
        <div className="max-w-3xl mx-auto px-8 py-5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-1.5 h-6 bg-gradient-to-b from-stone-800 to-stone-400 rounded-full" />
            <h1 className="text-lg font-medium tracking-tight text-stone-800">
              Model Parameters
            </h1>
          </div>
          <Link
            href="/ai-black-projects"
            className="text-sm text-stone-500 hover:text-stone-800 transition-colors flex items-center gap-1.5 group"
          >
            <span className="group-hover:-translate-x-0.5 transition-transform">←</span>
            <span>Back to simulation</span>
          </Link>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-3xl mx-auto px-8 py-12">
        {/* Hero Section */}
        <div className="mb-16">
          <p className="text-sm uppercase tracking-widest text-stone-400 mb-3 font-medium">
            Black Project Model
          </p>
          <h2 className="text-3xl font-semibold text-stone-900 tracking-tight mb-4" style={{ fontFamily: 'Georgia, serif' }}>
            Model Parameters
          </h2>
          <p className="text-stone-600 text-lg leading-relaxed max-w-2xl">
            What the parameters represent and how we chose their values.
          </p>
        </div>

        {/* Table of Contents - Refined */}
        <nav className="mb-20 pb-12 border-b border-stone-200/60">
          <p className="text-xs uppercase tracking-widest text-stone-400 mb-5 font-medium">
            Contents
          </p>
          <div className="grid grid-cols-2 gap-x-8 gap-y-3">
            {categoryEntries.map(([categoryId, category], index) => (
              <a
                key={categoryId}
                href={`#${categoryId}`}
                className="group flex items-baseline gap-3 text-stone-600 hover:text-stone-900 transition-colors"
              >
                <span className="text-xs text-stone-300 font-mono tabular-nums">
                  {String(index + 1).padStart(2, '0')}
                </span>
                <span className="text-[15px] group-hover:underline underline-offset-2">
                  {category.title}
                </span>
              </a>
            ))}
          </div>
        </nav>

        {/* Parameter Documentation Sections */}
        {categoryEntries.map(([categoryId, category], categoryIndex) => (
          <section key={categoryId} id={categoryId} className="mb-20">
            {/* Section Header */}
            <div className="flex items-start gap-4 mb-10">
              <span className="text-2xl text-stone-300 mt-1">
                {categoryIcons[categoryId] || '•'}
              </span>
              <div>
                <p className="text-xs uppercase tracking-widest text-stone-400 mb-1.5 font-medium">
                  Section {categoryIndex + 1}
                </p>
                <h2
                  className="text-2xl font-semibold text-stone-900 tracking-tight"
                  style={{ fontFamily: 'Georgia, serif' }}
                >
                  {category.title}
                </h2>
              </div>
            </div>

            {/* Parameter Articles */}
            <div className="space-y-0">
              {category.docs.map((doc, docIndex) => (
                <article
                  key={doc.filename}
                  className={`
                    relative pl-8
                    ${docIndex !== category.docs.length - 1 ? 'pb-12 border-l border-stone-200/60' : 'pb-4'}
                  `}
                >
                  {/* Timeline dot */}
                  <div className="absolute left-0 top-0 -translate-x-1/2 w-2 h-2 rounded-full bg-stone-300" />

                  {/* Parameter Name Title */}
                  <h3 className="text-lg font-mono text-stone-700 mb-4 tracking-tight">
                    {doc.paramName}
                  </h3>

                  {/* Prose Content */}
                  <div className="
                    prose prose-stone prose-sm max-w-none

                    [&>h1]:text-xl [&>h1]:font-semibold [&>h1]:text-stone-900
                    [&>h1]:mt-0 [&>h1]:mb-5 [&>h1]:tracking-tight
                    [&>h1]:font-[Georgia,serif]

                    [&>h2]:text-sm [&>h2]:font-semibold [&>h2]:text-stone-700
                    [&>h2]:mt-8 [&>h2]:mb-3 [&>h2]:uppercase [&>h2]:tracking-wide
                    [&>h2]:before:content-['—_'] [&>h2]:before:text-stone-300

                    [&>p]:text-stone-600 [&>p]:leading-[1.75] [&>p]:mb-4

                    [&>ul]:text-stone-600 [&>ul]:my-4 [&>ul]:ml-0 [&>ul]:list-none
                    [&>ul>li]:relative [&>ul>li]:pl-5 [&>ul>li]:mb-2
                    [&>ul>li]:before:content-[''] [&>ul>li]:before:absolute
                    [&>ul>li]:before:left-0 [&>ul>li]:before:top-[0.6em]
                    [&>ul>li]:before:w-1.5 [&>ul>li]:before:h-1.5
                    [&>ul>li]:before:bg-stone-300 [&>ul>li]:before:rounded-full

                    [&>ol]:text-stone-600 [&>ol]:my-4

                    [&_strong]:text-stone-800 [&_strong]:font-semibold

                    [&_code]:text-[13px] [&_code]:bg-stone-100/80
                    [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:rounded
                    [&_code]:font-mono [&_code]:text-stone-700
                    [&_code]:border [&_code]:border-stone-200/50

                    [&_pre]:bg-stone-50 [&_pre]:border [&_pre]:border-stone-200/60
                    [&_pre]:rounded-lg [&_pre]:p-4 [&_pre]:my-5
                    [&_pre]:overflow-x-auto
                    [&_pre_code]:bg-transparent [&_pre_code]:border-0
                    [&_pre_code]:p-0 [&_pre_code]:text-[13px]

                    [&_table]:w-full [&_table]:text-sm [&_table]:my-5
                    [&_table]:border-collapse

                    [&_thead]:border-b-2 [&_thead]:border-stone-200
                    [&_th]:px-4 [&_th]:py-3 [&_th]:text-left
                    [&_th]:font-semibold [&_th]:text-stone-700
                    [&_th]:text-xs [&_th]:uppercase [&_th]:tracking-wide

                    [&_tbody_tr]:border-b [&_tbody_tr]:border-stone-100
                    [&_tbody_tr:nth-child(even)]:bg-stone-50/50
                    [&_tbody_tr:hover]:bg-stone-100/50
                    [&_td]:px-4 [&_td]:py-3 [&_td]:text-stone-600

                    [&_img]:rounded-lg [&_img]:shadow-sm [&_img]:my-6
                    [&_img]:border [&_img]:border-stone-200/60

                    [&_a]:text-stone-700 [&_a]:underline [&_a]:underline-offset-2
                    [&_a]:decoration-stone-300 [&_a:hover]:decoration-stone-500
                    [&_a:hover]:text-stone-900

                    [&_blockquote]:border-l-2 [&_blockquote]:border-stone-300
                    [&_blockquote]:pl-4 [&_blockquote]:italic [&_blockquote]:text-stone-500
                  ">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{doc.content}</ReactMarkdown>
                  </div>
                </article>
              ))}
            </div>
          </section>
        ))}

        {/* Footer */}
        <footer className="mt-20 pt-10 border-t border-stone-200/60">
          <div className="flex items-center justify-between">
            <p className="text-sm text-stone-400">
              Black Project Simulation Model • AI Futures Simulator
            </p>
            <Link
              href="/ai-black-projects"
              className="text-sm text-stone-500 hover:text-stone-800 transition-colors flex items-center gap-1.5 group"
            >
              <span>Return to simulation</span>
              <span className="group-hover:translate-x-0.5 transition-transform">→</span>
            </Link>
          </div>
        </footer>
      </main>
    </div>
  );
}
