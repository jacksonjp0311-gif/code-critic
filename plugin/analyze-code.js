import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Code Feedback Neural Net — AGNT Plugin Tool
 *
 * Spawns a Python subprocess that runs the local neural network model
 * to analyze code quality. The Python side handles:
 *   1. Feature extraction (AST, tokenize, radon, telemetry)
 *   2. Neural network inference (compact transformer, ~1.2M params)
 *   3. Feedback generation (templates → natural language)
 *
 * All computation is local — no network calls, no API keys needed.
 */

// --------------------------------------------------------------------------- #
# Resolve paths to Python + model                                              #
# --------------------------------------------------------------------------- #
function findPython() {
  const candidates = ['python3', 'python', 'py'];
  // On Windows, 'python' usually works; on macOS/Linux, 'python3'
  for (const cmd of candidates) {
    try {
      const r = require('child_process').execSync(`${cmd} --version`, { stdio: 'pipe' });
      if (r && r.toString().includes('Python')) return cmd;
    } catch { /* not found */ }
  }
  return 'python'; // fallback
}

function findAnalyzeScript() {
  // The analyze.py lives alongside this JS file in the plugin folder
  const local = path.join(__dirname, 'analyze.py');
  if (fs.existsSync(local)) return local;

  // Fallback: check if it's in a parent directory (dev layout)
  const parent = path.resolve(__dirname, '..', 'analyze.py');
  if (fs.existsSync(parent)) return parent;

  return local; // will fail with a clear error
}

function findModel() {
  const candidates = [
    path.join(__dirname, 'code_feedback_model.pt'),
    path.join(__dirname, '..', 'code_feedback_model.pt'),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return candidates[0]; // fallback, will error clearly
}

// --------------------------------------------------------------------------- #
# Spawn Python subprocess                                                      #
# --------------------------------------------------------------------------- #
function runPythonAnalysis(code, filePath, telemetry, timeoutMs = 30000) {
  return new Promise((resolve) => {
    const scriptPath = findAnalyzeScript();
    const modelPath = findModel();

    if (!fs.existsSync(scriptPath)) {
      resolve({
        quality_score: 0, quality_label: 'Error', confidence: 0,
        issues: '[]', suggestions: '[]', positive_notes: '[]',
        feedback_text: `❌ analyze.py not found at ${scriptPath}`,
        inference_time_ms: 0,
        error: `analyze.py not found at ${scriptPath}`,
      });
      return;
    }

    const args = [scriptPath, '--code', code, '--json', '--model', modelPath];
    if (filePath) args.push('--file', filePath);
    if (telemetry) args.push('--telemetry', typeof telemetry === 'string' ? telemetry : JSON.stringify(telemetry));

    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    const child = spawn(pythonCmd, args, {
      cwd: __dirname,
      env: { ...process.env },
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    let done = false;

    const finish = (result) => {
      if (!done) { done = true; resolve(result); }
    };

    const timer = setTimeout(() => {
      child.kill();
      finish({
        quality_score: 0, quality_label: 'Timeout', confidence: 0,
        issues: '[]', suggestions: '[]', positive_notes: '[]',
        feedback_text: `⏱ Analysis timed out after ${timeoutMs}ms`,
        inference_time_ms: timeoutMs,
        error: `Python process timed out after ${timeoutMs}ms`,
      });
    }, timeoutMs);

    child.stdout.on('data', (d) => { stdout += d.toString(); });
    child.stderr.on('data', (d) => { stderr += d.toString(); });
    child.on('close', (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        finish({
          quality_score: 0, quality_label: 'Error', confidence: 0,
          issues: '[]', suggestions: '[]', positive_notes: '[]',
          feedback_text: `❌ Python exited with code ${code}\n${stderr}`,
          inference_time_ms: 0,
          error: `Python exited with code ${code}: ${stderr.slice(0, 500)}`,
        });
      } else {
        try {
          const parsed = JSON.parse(stdout.trim());
          finish({
            quality_score: parsed.quality_score ?? 0,
            quality_label: parsed.quality_label ?? 'Unknown',
            quality_emoji: parsed.quality_emoji ?? '',
            confidence: parsed.confidence ?? 0,
            issues: JSON.stringify(parsed.issues ?? []),
            suggestions: JSON.stringify(parsed.suggestions ?? []),
            positive_notes: JSON.stringify(parsed.positive_notes ?? []),
            feedback_text: parsed.feedback_text ?? stdout.trim(),
            inference_time_ms: parsed.inference_time_ms ?? 0,
            error: parsed.error ?? null,
          });
        } catch {
          finish({
            quality_score: 0, quality_label: 'Parse Error', confidence: 0,
            issues: '[]', suggestions: '[]', positive_notes: '[]',
            feedback_text: stdout.trim() || 'No output from analyzer',
            inference_time_ms: 0,
            error: `Failed to parse JSON output: ${stdout.slice(0, 200)}`,
          });
        }
      }
    });
    child.on('error', (err) => {
      clearTimeout(timer);
      finish({
        quality_score: 0, quality_label: 'Error', confidence: 0,
        issues: '[]', suggestions: '[]', positive_notes: '[]',
        feedback_text: `❌ Failed to spawn Python: ${err.message}`,
        inference_time_ms: 0,
        error: `Failed to spawn Python: ${err.message}`,
      });
    });
  });
}

// --------------------------------------------------------------------------- #
# Plugin tool class                                                            #
# --------------------------------------------------------------------------- #
class AnalyzeCodeTool {
  constructor() {
    this.name = 'analyze-code';
  }

  async execute(params, inputData, workflowEngine) {
    const startTime = Date.now();

    try {
      // Extract parameters
      const code = params?.code ?? inputData?.code ?? '';
      const filePath = params?.filePath ?? inputData?.filePath ?? null;
      let telemetry = params?.telemetry ?? inputData?.telemetry ?? null;

      // Validate
      if (!code || typeof code !== 'string' || code.trim().length === 0) {
        return {
          quality_score: 0,
          quality_label: 'Error',
          quality_emoji: '',
          confidence: 0,
          issues: '[]',
          suggestions: '[]',
          positive_notes: '[]',
          feedback_text: '❌ No code provided. Pass a code string to analyze.',
          inference_time_ms: 0,
          error: 'Missing required parameter: code (string)',
        };
      }

      // Parse telemetry if it's a string
      if (typeof telemetry === 'string') {
        try { telemetry = JSON.parse(telemetry); } catch { telemetry = null; }
      }

      // Limit code size to prevent OOM / timeouts
      const maxChars = 50000;
      const trimmedCode = code.length > maxChars ? code.slice(0, maxChars) : code;

      // Run Python analysis
      const result = await runPythonAnalysis(trimmedCode, filePath, telemetry);
      result.inference_time_ms = (Date.now() - startTime);

      return result;
    } catch (error) {
      console.error('[analyze-code] Error:', error);
      return {
        quality_score: 0,
        quality_label: 'Error',
        quality_emoji: '',
        confidence: 0,
        issues: '[]',
        suggestions: '[]',
        positive_notes: '[]',
        feedback_text: `❌ Analysis failed: ${error.message}`,
        inference_time_ms: Date.now() - startTime,
        error: error.message,
      };
    }
  }
}

export default new AnalyzeCodeTool();
