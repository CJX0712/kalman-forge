/* UI 冒烟：用最小 DOM stub 在 Node 里跑 <script id="ui">，抓只有浏览器才会炸的运行时错误。
   用法: node _uicheck.js */
'use strict';
const fs = require('fs'), vm = require('vm'), path = require('path');
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const eng = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const ui = html.match(/<script id="ui">([\s\S]*?)<\/script>/);
if (!eng || !ui) { console.error('找不到 engine / ui 脚本块'); process.exit(1); }

/* ---- canvas 2d context stub：任何方法都 no-op，属性可读写 ---- */
function ctx2d() {
  return new Proxy({}, {
    get(t, k) {
      if (k in t) return t[k];
      return function () { };
    },
    set(t, k, v) { t[k] = v; return true; }
  });
}
function makeEl(id, isCanvas) {
  const el = {
    id, value: id === 'inp-q' ? '-1' : id === 'inp-r' ? '0' : id === 'inp-mis' ? '0' : 'cv',
    textContent: '', innerHTML: '', clientWidth: 620, width: 0, height: 0,
    style: {}, disabled: false,
    getContext: () => ctx2d(),
    addEventListener() { }
  };
  if (id === 'sel-scene') el.value = 'cv';
  return el;
}
const els = {};
function $(id) { if (!els[id]) els[id] = makeEl(id, id.startsWith('cv-')); return els[id]; }

const errors = [];
const alerts = [];
const ctx = {
  console, Math, JSON, Array, Object, Number, String, Boolean, Error, isFinite, isNaN, Infinity, NaN,
  Float64Array, Uint8Array, TextEncoder, TextDecoder, setTimeout: (fn) => fn(),
  document: { getElementById: $ },
  window: { devicePixelRatio: 1, addEventListener() { } },
  alert: (s) => alerts.push(s)
};
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(eng[1], ctx, { filename: 'engine.js' });
if (!ctx.KF) { console.error('engine 未暴露 KF'); process.exit(1); }

let uiRan = false;
try {
  vm.runInContext(ui[1], ctx, { filename: 'ui.js' });
  uiRan = true;
} catch (e) { errors.push('初始渲染: ' + e.message + '\n' + (e.stack || '').split('\n').slice(0, 4).join('\n')); }

function fire(name, fn) { try { fn(); } catch (e) { errors.push(name + ': ' + e.message + '\n' + (e.stack || '').split('\n').slice(0, 4).join('\n')); } }

if (uiRan) {
  /* 切换场景 */
  ['rw', 'radar', 'cv'].forEach(sc => {
    fire('切换场景=' + sc, () => { $('sel-scene').value = sc; if ($('sel-scene').onchange) $('sel-scene').onchange(); });
  });
  /* 拖动滑块 */
  [['inp-q', '0.4'], ['inp-r', '-1'], ['inp-mis', '0.8'], ['inp-mis', '-0.7']].forEach(([id, v]) => {
    fire('滑块 ' + id + '=' + v, () => { $(id).value = v; if ($(id).oninput) $(id).oninput(); });
  });
  /* 重新采样 */
  fire('重新采样', () => { if ($('btn-run').onclick) $('btn-run').onclick(); });
  /* EM */
  fire('EM 学习', () => { if ($('btn-em').onclick) $('btn-em').onclick(); });
  /* 自检 */
  fire('运行自检', () => { if ($('btn-test').onclick) $('btn-test').onclick(); });
}

console.log('UI 冒烟：' + (errors.length ? 'FAILED' : 'OK'));
if (!uiRan) console.log('  UI 脚本未能执行');
console.log('  场景切换/滑块/重采样/EM/自检 均已触发');
console.log('  alert 次数 = ' + alerts.length + (alerts.length ? '（首条：' + String(alerts[0]).split('\n')[0] + '）' : ''));
console.log('  自检表格是否写入 = ' + (String($('tests').innerHTML).indexOf('PASS') >= 0 ||
  String($('tests').innerHTML).indexOf('FAIL') >= 0));
console.log('  统计卡是否写入 = ' + (String($('stats').innerHTML).indexOf('RMSE') >= 0));
errors.forEach(e => console.log('\n✗ ' + e));
process.exitCode = errors.length ? 1 : 0;
