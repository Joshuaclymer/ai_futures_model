# Repository Guidelines

## Project Structure & Module Organization
The Next.js UI lives under `app/`, where route segments map to pages; shared client/server components sit in `components/`, and reusable hooks/utilities live in `lib/`. Configuration and domain constants are centralized in `config/` and `constants/`. The Flask service that feeds the charts is defined in `api/`, pulling default inputs from `data/` and `inputs/`. Generated artifacts such as `diagram.svg` and static assets belong in `public/` and `svgs/`. Update `benchmark_results.yaml` when benchmarks or chart labels change.

## Build, Test, and Development Commands
Run `npm run dev` to launch Next.js (Turbopack) alongside the Flask API; use `npm run next-dev` or `npm run flask-dev` if you need a single stack. Build production assets with `npm run build`, and start the compiled app using `npm run start`. Regenerate architecture diagrams with `npm run mermaid`. Always install Python dependencies before backend work: `pip3 install -r requirements.txt`.

## Coding Style & Naming Conventions
TypeScript modules use two-space indentation, ES modules, and `PascalCase` filenames for components (e.g., `ProgressChartServer.tsx`), while helper functions remain `camelCase`. Favor server components for data preparation and keep client-only logic in `ProgressChartClient`. Tailwind utility classes drive layout; keep theme tokens in `globals.css` current when altering colors. Python modules follow PEP 8 and `snake_case`; log meaningful warnings rather than raising for optional data.

Avoid using type casts if at all possible.  (Explicit annotations, i.e. `const foo: TypeName = ...'` are fine but optional.)  If you absolutely must apply a type cast somewhere, always leave the following comment above it:
```typescript
// TODO: I AM AN INSTANCE OF ${MODEL_NAME} AND HAVE APPLIED A TYPE CAST HERE BECAUSE I COULDN'T MAKE IT WORK OTHERWISE, PLEASE FIX THIS
```

Avoid leaving spurious or trivial comments that can easily be inferred from the code they're documenting.

## Testing Guidelines
Linting is the minimum bar: run `npm run lint` before every commit. When adding UI logic, create React tests under `components/__tests__/` using Testing Library patterns (`render`, `screen`). New Python model routines should include unit tests in `api/tests/` leveraging `pytest` and lightweight fixtures that load from `input_data.csv`. Document any manual verification steps for Monte Carlo outputs within your PR description.

## Commit & Pull Request Guidelines
Recent history favors short, imperative titles (`Refactor`, `Diagram tweak`). Follow that style and scope one logical change per commit. Pull requests must describe the user-visible impact, list primary commands executed, and link related issues. Attach updated screenshots or chart snippets when visuals change, and mention if `benchmark_results.yaml` or generated diagrams were touched.

## Data & Automation Notes
Model parameters live in `model_config.py` and `progress_model.py`; update both when adding new sliders or plot metadata. Keep large CSVs out of version control—drop samples in `inputs/` and update `.gitignore` when necessary. If an API change alters the Monte Carlo blueprint, ensure the Flask blueprint import still succeeds with graceful fallbacks.
