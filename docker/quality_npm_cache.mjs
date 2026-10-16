// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Build step of gbench's quality image (docker/quality.Dockerfile).
//
// When GemmaClaw loads a bundled plugin it installs the plugin's runtime dependencies with
// `npm install --ignore-scripts <name>@<spec> ...` (src/plugins/bundled-runtime-deps.ts: the
// `dependencies` and `optionalDependencies` of the plugin's package.json, `workspace:` ones left
// out). The quality container has no network, so this fills an npm cache with every such
// dependency of every bundled plugin (extensions/, dist/extensions/, dist-runtime/extensions/),
// transitive ones included; the run points npm at a copy of it, offline. A name with more than
// one spec gets one install per spec.
//
// Usage: node quality_npm_cache.mjs <gemmaclaw root> <npm cache dir>
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const [root, cache] = process.argv.slice(2);
if (!root || !cache) {
  console.error("usage: node quality_npm_cache.mjs <gemmaclaw root> <npm cache dir>");
  process.exit(2);
}

const specs = new Map();
for (const dir of ["extensions", "dist/extensions", "dist-runtime/extensions"]) {
  const base = path.join(root, dir);
  if (!fs.existsSync(base)) {
    continue;
  }
  for (const id of fs.readdirSync(base).sort()) {
    const manifest = path.join(base, id, "package.json");
    if (!fs.existsSync(manifest)) {
      continue;
    }
    const pkg = JSON.parse(fs.readFileSync(manifest, "utf8"));
    const deps = { ...(pkg.dependencies ?? {}), ...(pkg.optionalDependencies ?? {}) };
    for (const [name, raw] of Object.entries(deps)) {
      const spec = typeof raw === "string" ? raw.trim() : "";
      if (!spec || spec.toLowerCase().startsWith("workspace:")) {
        continue;
      }
      if (!specs.has(name)) {
        specs.set(name, new Set());
      }
      specs.get(name).add(spec);
    }
  }
}

const rounds = [];
for (const name of [...specs.keys()].sort()) {
  [...specs.get(name)].sort().forEach((spec, i) => {
    (rounds[i] ??= {})[name] = spec;
  });
}
// Each round is installed twice: from the registry into the cache, then with --offline from the
// cache alone, which proves the cache holds everything it needs.
for (const offline of [false, true]) {
  rounds.forEach((dependencies, i) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gbench-npm-cache-"));
    fs.writeFileSync(
      path.join(dir, "package.json"),
      JSON.stringify({ name: `gbench-plugin-runtime-deps-${i}`, private: true, dependencies }),
    );
    execFileSync(
      "npm",
      ["install", "--ignore-scripts", "--legacy-peer-deps", "--no-audit", "--no-fund",
       "--cache", cache, ...(offline ? ["--offline"] : [])],
      { cwd: dir, stdio: "inherit" },
    );
    fs.rmSync(dir, { recursive: true, force: true });
  });
}
console.log(
  `gbench: cached the runtime dependencies of GemmaClaw's bundled plugins ` +
    `(${specs.size} packages, ${rounds.length} install(s)) in ${cache}`,
);
