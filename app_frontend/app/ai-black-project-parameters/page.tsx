import Link from 'next/link';
import fs from 'fs';
import path from 'path';
import ReactMarkdown from 'react-markdown';

export const metadata = {
  title: 'Black Project Parameters Documentation',
  description: 'Documentation for all parameters in the Black Project simulation model',
};

// Parameter documentation categories
const parameterCategories: Record<string, { title: string; files: string[] }> = {
  'detection': {
    title: 'Detection Parameters',
    files: [
      'detection_time.md',
      'chip_stock_detection.md',
      'energy_accounting_detection.md',
      'satellite_datacenter_detection.md',
      'sme_inventory_detection.md',
      'covert_unconcealed.md',
      'prior_odds.md',
    ],
  },
  'chips-and-fabs': {
    title: 'Chips and Fabs',
    files: [
      'fraction_diverted.md',
      'chips_per_wafer.md',
      'fab_construction_time.md',
      'operating_labor_production.md',
      'scanner_production_capacity.md',
      'prc_scanner_rampup.md',
      'prc_sme_indigenization.md',
      'retrofitted_capacity.md',
      'transistor_density.md',
      'ai_chip_lifespan.md',
      'prc_capacity.md',
    ],
  },
  'datacenters-and-energy': {
    title: 'Datacenters and Energy',
    files: [
      'datacenter_start_year.md',
      'construction_workers.md',
      'prc_energy.md',
      'prc_energy_consumption.md',
      'mw_per_worker.md',
      'h100_power.md',
      'energy_efficiency.md',
      'energy_efficiency_improvement.md',
      'max_energy_proportion.md',
    ],
  },
  'compute-trends': {
    title: 'Compute Trends',
    files: [
      'architecture_efficiency.md',
      'dennard_scaling_end.md',
      'watts_per_tpp.md',
      'watts_per_tpp_after_dennard.md',
      'watts_per_tpp_before_dennard.md',
      'largest_ai_project.md',
    ],
  },
  'other': {
    title: 'Other Parameters',
    files: [
      'project_property.md',
    ],
  },
};

// Read markdown files from the parameter_documentation folder
function getParameterDocs(): Record<string, { title: string; docs: { filename: string; content: string }[] }> {
  const docsPath = path.join(process.cwd(), '..', 'ai_futures_simulator', 'parameters', 'parameter_documentation');

  const categories: Record<string, { title: string; docs: { filename: string; content: string }[] }> = {};

  for (const [categoryId, category] of Object.entries(parameterCategories)) {
    const docs: { filename: string; content: string }[] = [];

    for (const filename of category.files) {
      const filePath = path.join(docsPath, filename);
      try {
        if (fs.existsSync(filePath)) {
          const content = fs.readFileSync(filePath, 'utf-8');
          docs.push({ filename, content });
        }
      } catch {
        // File doesn't exist, skip it
      }
    }

    if (docs.length > 0) {
      categories[categoryId] = {
        title: category.title,
        docs,
      };
    }
  }

  return categories;
}

export default function BlackProjectParametersPage() {
  const categories = getParameterDocs();

  return (
    <div className="min-h-screen bg-[#fffff8]">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <h1 className="text-xl font-bold text-gray-800">Black Project Parameters</h1>
          <Link
            href="/ai-black-projects"
            className="text-sm text-[#5E6FB8] hover:text-[#4B5A93] hover:underline"
          >
            ← Back to simulation
          </Link>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-6 py-8">
        <p className="text-gray-600 mb-8">
          This page documents the parameters used in the Black Project simulation model.
        </p>

        {/* Table of Contents */}
        <nav className="bg-white border border-gray-200 rounded-lg p-6 mb-10">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Table of Contents</h2>
          <ul className="space-y-2 text-sm">
            {Object.entries(categories).map(([categoryId, category]) => (
              <li key={categoryId}>
                <a href={`#${categoryId}`} className="text-[#5E6FB8] hover:underline">
                  {category.title}
                </a>
              </li>
            ))}
          </ul>
        </nav>

        {/* Parameter Documentation Sections */}
        {Object.entries(categories).map(([categoryId, category]) => (
          <section key={categoryId} id={categoryId} className="mb-12">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 pb-2 border-b border-gray-200">
              {category.title}
            </h2>

            <div className="space-y-6">
              {category.docs.map((doc) => (
                <div
                  key={doc.filename}
                  className="bg-white border border-gray-200 rounded-lg p-5 prose prose-sm max-w-none prose-headings:mt-4 prose-headings:mb-2 prose-h1:text-lg prose-h1:font-semibold prose-h1:text-gray-800 prose-h2:text-base prose-h2:font-medium prose-h2:text-gray-700 prose-p:text-gray-600 prose-p:leading-relaxed prose-ul:text-gray-600 prose-code:text-sm prose-code:bg-gray-100 prose-code:px-1 prose-code:rounded"
                >
                  <ReactMarkdown>{doc.content}</ReactMarkdown>
                </div>
              ))}
            </div>
          </section>
        ))}

        {/* Footer */}
        <footer className="mt-16 pt-8 border-t border-gray-200 text-center text-sm text-gray-500">
          <p>
            This documentation is for the Black Project simulation model in the AI Futures Simulator.
          </p>
          <p className="mt-2">
            <Link href="/ai-black-projects" className="text-[#5E6FB8] hover:underline">
              Return to simulation →
            </Link>
          </p>
        </footer>
      </main>
    </div>
  );
}
