/**
 * OpenClaw Internal Hook — Post-Session Cognitive Compression
 *
 * Triggers when a new session starts, compressing the PREVIOUS session
 * through Nova's cognitive deduplication pipeline. The more sessions
 * flow through, the better compression gets.
 *
 * Install: Add to ~/.openclaw/openclaw.json hooks.internal.entries
 */
import { execFile } from 'node:child_process';
import { access, constants } from 'node:fs/promises';
import { resolve } from 'node:path';
import { homedir } from 'node:os';

const NOVA_DIR = resolve(homedir(), 'Nova-v1');
const PYTHON = resolve(NOVA_DIR, '.venv/bin/python');
const HOOK_SCRIPT = resolve(NOVA_DIR, 'scripts/openclaw_session_hook.py');
const OUTPUT_DIR = resolve(homedir(), '.openclaw/compressed');

async function fileExists(path) {
  try {
    await access(path, constants.F_OK);
    return true;
  } catch { return false; }
}

const handler = async (event) => {
  // Only trigger on /new command (previous session ending)
  if (event.type !== 'command' || event.action !== 'new') return;

  const prevFile = event.context?.previousSessionEntry?.sessionFile;
  if (!prevFile) return;

  // Verify the session file exists
  if (!await fileExists(prevFile)) {
    console.log('[nova-compress] Session file not found:', prevFile);
    return;
  }

  console.log('[nova-compress] Compressing previous session:', prevFile);

  // Run the Python compression script in background (don't block the gateway)
  try {
    const child = execFile(PYTHON, [HOOK_SCRIPT, prevFile, '--output-dir', OUTPUT_DIR], {
      timeout: 120000,  // 2 min max
      env: { ...process.env, C3AE_DATA_DIR: resolve(homedir(), 'Nova-v1/data') },
    }, (error, stdout, stderr) => {
      if (error) {
        console.error('[nova-compress] Error:', error.message);
        return;
      }
      if (stdout) console.log('[nova-compress]', stdout.trim());
      if (stderr) console.error('[nova-compress]', stderr.trim());
    });
  } catch (err) {
    console.error('[nova-compress] Failed to spawn:', err.message);
  }
};

export default handler;
