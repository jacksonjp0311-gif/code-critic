import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Code Critic v2 — AGNT Plugin Tool
 *
 * Uses the improved CodeCriticV2 model with:
 * - Code-aware tokenizer (465 vocab, sees actual code tokens)
 * - Dual-input architecture (token embeddings + structural features)
 * - Trained on 5000 labeled samples across 6 issue categories
 * - 1.31M params, 2.51 MB model file
 */

function findScript() {
  const local = path.join(__dirname, 'analyze_v2.py');
  if (fs.existsSync(local)) return local;
  const parent = path.resolve(__dirname, '..', 'analyze_v2.py');
  if (fs.existsSync(parent)) return parent;
  // Fallback to v1
  const v1 = path.join(__dirname, 'analyze.py');
  if (fs.existsSync(v1)) return v1;
  return local;
}

function findModel() {
  const candidates = [
    path.join(__dirname, 'code_critic_v2.pt'),
    path.join(__dirname, '..', 'code_critic_v2.pt'),
    path.join(__dirname, 'code_feedback_model.pt'),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return candidates[0];
}

function runAnalysis(code, filePath, telemetry, timeoutMs = 30000) {
  return new Promise((resolve) => {
    const scriptPath = findScript();
    const modelPath = findModel();

    if (!fs.existsSync(scriptPath)) {
      resolve({
        success: false, quality_score: 0, quality_label: 'Error', confidence: 0,
        issues: '[]', suggestions: '[]', positive_notes: '[]',
        feedback_text: '❌ analyze script not found: ' + scriptPath,
        inference_time_ms: 0, error: 'Script not found: ' + scriptPath,
      });
      return;
    }

    const args = [scriptPath, '--code', code, '--json', '--model', modelPath];
    if (filePath) args.push('--file', filePath);
    if (telemetry) {
      args.push('--telemetry', typeof telemetry === 'string' ? telemetry : JSON.stringify(telemetry));
    }

    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    const child = spawn(pythonCmd, args, {
      cwd: __dirname, env: { ...process.env }, stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '', stderr = '', done = false;
    const finish = (r) => { if (!done) { done = true; resolve(r); } };

    const timer = setTimeout(() => {
      child.kill();
      finish({ success: false, quality_score: 0, quality_label: 'Timeout', confidence: 0,
        issues: '[]', suggestions: '[]', positive_notes: '[]',
        feedback_text: '⏱ Timeout after ' + timeoutMs + 'ms',
        inference_time_ms: timeoutMs, error: 'Timeout' });
    }, timeoutMs);

    child.stdout.on('data', (d) => { stdout += d.toString(); });
    child.stderr.on('data', (d) => { stderr += d.toString(); });
    child.on('close', (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        finish({ success: false, quality_score: 0, quality_label: 'Error', confidence: 0,
          issues: '[]', suggestions: '[]', positive_notes: '[]',
          feedback_text: '❌ Python exit code ' + code + '\n' + stderr,
          inference_time_ms: 0, error: 'Exit ' + code + ': ' + stderr.slice(0, 500) });
      } else {
        try {
          const p = JSON.parse(stdout.trim());
          finish({ success: true,
            quality_score: p.quality_score ?? 0, quality_label: p.quality_label ?? 'Unknown',
            quality_emoji: p.quality_emoji ?? '', confidence: p.confidence ?? 0,
            issues: JSON.stringify(p.issues ?? []), suggestions: JSON.stringify(p.suggestions ?? []),
            positive_notes: JSON.stringify(p.positive_notes ?? []),
            feedback_text: p.feedback_text ?? stdout.trim(),
            inference_time_ms: p.inference_time_ms ?? 0, error: p.error ?? null });
        } catch {
          finish({ success: false, quality_score: 0, quality_label: 'Parse Error', confidence: 0,
            issues: '[]', suggestions: '[]', positive_notes: '[]',
            feedback_text: stdout.trim() || 'No output', inference_time_ms: 0,
            error: 'JSON parse failed: ' + stdout.slice(0, 200) });
        }
      }
    });
    child.on('error', (err) => {
      clearTimeout(timer);
      finish({ success: false, quality_score: 0, quality_label: 'Error', confidence: 0,
        issues: '[]', suggestions: '[]', positive_notes: '[]',
        feedback_text: '❌ Spawn error: ' + err.message, inference_time_ms: 0, error: err.message });
    });
  });
}

class AnalyzeCode {
  constructor() { this.name = 'analyze-code'; }

  async execute(params) {
    const code = params?.code ?? '';
    const filePath = params?.filePath ?? null;
    let telemetry = params?.telemetry ?? null;

    if (!code || typeof code !== 'string' || code.trim().length === 0) {
      return { success: false, quality_score: 0, quality_label: 'Error', quality_emoji: '',
        confidence: 0, issues: '[]', suggestions: '[]', positive_notes: '[]',
        feedback_text: '❌ No code provided.', inference_time_ms: 0, error: 'Missing: code' };
    }
    if (typeof telemetry === 'string') {
      try { telemetry = JSON.parse(telemetry); } catch { telemetry = null; }
    }
    const maxChars = 50000;
    const trimmed = code.length > maxChars ? code.slice(0, maxChars) : code;
    try {
      return await runAnalysis(trimmed, filePath, telemetry);
    } catch (err) {
      return { success: false, quality_score: 0, quality_label: 'Error', quality_emoji: '',
        confidence: 0, issues: '[]', suggestions: '[]', positive_notes: '[]',
        feedback_text: '❌ ' + err.message, inference_time_ms: 0, error: err.message };
    }
  }
}

export default new AnalyzeCode();
