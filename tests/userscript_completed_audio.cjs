/* Exercise the shipped final-track lifecycle without models, network or media files. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const start = source.indexOf('  async function attachAudio(');
const end = source.indexOf('  function mediaUrlOf(', start);
assert(start > 0 && end > start, 'completed audio lifecycle exists');
const lifecycle = source.slice(start, end);
let passed = 0;

function fixture({ constructError = false, playError = null } = {}) {
  const state = new WeakMap();
  const events = new Map();
  const video = {
    currentTime: 3, paused: false, playbackRate: 1.25, muted: false,
    addEventListener: (name, fn) => events.set(name, fn),
    removeEventListener: (name, fn) => { if (events.get(name) === fn) events.delete(name); },
  };
  const s = { cancelled: false, progressive: { active: true } };
  state.set(video, s);
  const calls = [], created = [], revoked = [], sounds = [];
  let resolve, reject, releases = 0, errors = 0, retries = 0, stoppedProgressive = 0;
  const timerIds = new Set();
  const context = {
    state,
    apiAudio(url, options, server) {
      calls.push({ url, options, server });
      return new Promise((res, rej) => { resolve = res; reject = rej; });
    },
    URL: {
      createObjectURL(blob) { assert.equal(blob, 'encoded-m4a'); const url = `blob:audio-${created.length}`; created.push(url); return url; },
      revokeObjectURL(url) { revoked.push(url); },
    },
    Audio: class {
      constructor(src) {
        if (constructError) throw new Error('audio construction failed');
        this.src = src; this.currentTime = 0; this.listeners = new Map(); sounds.push(this);
      }
      addEventListener(name, fn) { this.listeners.set(name, fn); }
      removeEventListener(name, fn) { if (this.listeners.get(name) === fn) this.listeners.delete(name); }
      pause() { this.paused = true; }
      play() { this.paused = false; return playError ? Promise.reject(playError) : Promise.resolve(); }
    },
    detachProgressive(target) { assert.equal(target, video); if (s.progressive) stoppedProgressive++; s.progressive = null; },
    createDucker() { return { set(value) { s.duckValue = value; }, release() { releases++; } }; },
    buildWindows: entries => entries,
    inWindow: () => true,
    prefs: { voiceVol: .7, duck: .15 },
    DRIFT_S: .12, SEEK_BIAS_S: .05,
    setInterval: fn => { timerIds.add(fn); return fn; },
    clearInterval: id => timerIds.delete(id),
    setRetryButton: () => retries++, showError: () => errors++,
  };
  vm.createContext(context);
  vm.runInContext(lifecycle, context);
  return {
    context, state, video, s, calls, created, revoked, sounds, events, timerIds,
    resolve: () => resolve('encoded-m4a'), reject: error => reject(error),
    counts: () => ({ releases, errors, retries, stoppedProgressive }),
  };
}

(async () => {
  {
    const f = fixture();
    const abort = new AbortController();
    const server = { url: 'http://127.0.0.1:8765' };
    const ready = f.context.attachAudio(f.video, '/audio/job.m4a?access=test', [], server, abort.signal);
    assert.equal(f.s.progressive.active, true, 'keep clips playing while complete audio downloads');
    assert.equal(f.calls[0].options.signal, abort.signal);
    assert.equal(f.calls[0].url, '/audio/job.m4a?access=test');
    assert.equal(f.calls[0].server, server);
    assert.deepEqual(f.created, []);
    f.resolve();
    assert.equal(await ready, true);
    const audio = f.s.audio;
    assert.equal(audio.src, 'blob:audio-0', 'HTMLAudio never connects directly to localhost');
    assert.equal(audio.currentTime, 3);
    assert.equal(audio.playbackRate, 1.25);
    assert.equal(audio.volume, .7);
    assert.equal(f.counts().stoppedProgressive, 1);
    f.video.currentTime = 14; f.events.get('seeked')();
    assert.equal(audio.currentTime, 14.05);
    f.context.detachAudio(f.video);
    assert.deepEqual(f.revoked, f.created);
    assert.equal(f.events.size, 0);
    assert.equal(f.timerIds.size, 0);
    assert.equal(audio.src, '');
    assert.equal(f.s.audio, null);
    assert.equal(f.counts().releases, 1);
    f.context.detachAudio(f.video);
    assert.equal(f.revoked.length, 1, 'cleanup is idempotent');
    passed++;
  }
  for (const reason of ['aborted', 'cancelled', 'removed', 'replaced']) {
    const f = fixture(); const abort = new AbortController();
    const ready = f.context.attachAudio(f.video, '/audio/job.m4a', [], {}, abort.signal);
    if (reason === 'aborted') abort.abort();
    if (reason === 'cancelled') f.s.cancelled = true;
    if (reason === 'removed') f.state.delete(f.video);
    if (reason === 'replaced') f.state.set(f.video, {});
    f.resolve();
    assert.equal(await ready, false, reason);
    assert.deepEqual(f.created, [], reason + ' does not allocate or attach stale sound');
    assert.equal(f.counts().stoppedProgressive, 0);
    passed++;
  }
  {
    const f = fixture(); const abort = new AbortController(); abort.abort();
    assert.equal(await f.context.attachAudio(f.video, '/audio/x', [], {}, abort.signal), false);
    assert.equal(f.calls.length, 0); passed++;
  }
  {
    const f = fixture(); const ready = f.context.attachAudio(f.video, '/audio/x', [], {});
    f.reject(new Error('transport failed'));
    await assert.rejects(ready, /transport failed/);
    assert.deepEqual(f.created, []);
    assert.equal(f.s.progressive.active, true); passed++;
  }
  {
    const f = fixture({ constructError: true }); const ready = f.context.attachAudio(f.video, '/audio/x', [], {});
    f.resolve(); await assert.rejects(ready, /construction failed/);
    assert.deepEqual(f.revoked, f.created);
    assert.equal(f.s.progressive.active, true); passed++;
  }
  {
    const f = fixture(); const ready = f.context.attachAudio(f.video, '/audio/x', [], {});
    f.resolve(); await ready;
    f.s.audio.listeners.get('error')();
    assert.deepEqual(f.revoked, f.created);
    assert.equal(f.s.on, false);
    assert.equal(f.counts().errors, 1);
    assert.equal(f.counts().retries, 1);
    assert.equal(f.counts().releases, 1); passed++;
  }
  {
    const error = new Error('CSP rejected audio'); error.name = 'NotSupportedError';
    const f = fixture({ playError: error });
    const ready = f.context.attachAudio(f.video, '/audio/x', [], {});
    f.resolve(); await ready; await Promise.resolve();
    assert.deepEqual(f.revoked, f.created);
    assert.equal(f.counts().errors, 1);
    assert.equal(f.s.on, false); passed++;
  }
  process.stdout.write(JSON.stringify({ passed }) + '\n');
})().catch(error => { process.stderr.write(error.stack + '\n'); process.exitCode = 1; });
