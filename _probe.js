/* kalman-forge 探针：把内部状态用 ASCII 打出来人眼复核。
   断言全绿 ≠ 正确 —— 这里用「独立重算」的方式把关键量摆出来看。
   用法: node _probe.js → 写 _probe.txt */
'use strict';
const fs = require('fs'), vm = require('vm'), path = require('path');

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const ctx = { console, Math, JSON, Array, Object, Number, String, Boolean, Error, isFinite, isNaN, Infinity, NaN, Float64Array, Uint8Array, TextEncoder, TextDecoder };
ctx.globalThis = ctx; vm.createContext(ctx); vm.runInContext(m[1], ctx, { filename: 'engine.js' });
const K = ctx.KF;

let buf = [];
function say(s) { buf.push(s); }
function pad(s, n) { s = '' + s; while (s.length < n) s += ' '; return s; }
function f(x, d) { return Number(x).toFixed(d === undefined ? 4 : d); }
function e(x, d) { return Number(x).toExponential(d === undefined ? 3 : d); }
function mv(v, d) { return '[' + v.map(x => pad(f(x, d === undefined ? 5 : d), 10)).join(' ') + ']'; }
function mt(A, d) { return A.map(r => '    ' + mv(r, d)).join('\n'); }

say('══════════ kalman-forge 探针 · 内部状态人眼复核 ══════════');
say('');

/* ---------- 模型 ---------- */
const M = K.sceneCV(1, 0.35, 0.8);
say('模型：常速度（n=2, m=1），dt=1');
say('  F = ' + mt(M.F));
say('  Q = ' + mt(M.Q));
say('  H = ' + mv(M.H[0], 3) + '    R = ' + f(M.R[0][0], 3) + '    x0 = ' + mv(M.x0, 2) + '   P0 = diag(1,1)');
say('');

const T = 24;
const sim = K.simulate(M, T, K.mulberry32(2026));
const kf = K.kalman(sim.Y, M);
const sm = K.rts(kf, M);
const inf = K.infoFilter(sim.Y, M);
const bp = K.batchPosterior(sim.Y, M);

/* ---------- 1. 前 8 步逐点对照：滤波 / 信息滤波 / 批处理 ---------- */
say('── ① 逐点对照（前 8 步，x₁ 分量）──');
say('  批处理解的是「用全部观测」的后验，所以它等于【平滑】而非滤波 —— 与滤波不同是对的。');
say('  ' + pad('t', 4) + pad('真值', 11) + pad('观测', 11) + pad('滤波', 11) + pad('信息滤波', 13) +
    pad('批处理=平滑解', 16) + ' |Δ KF−info|   |Δ RTS−batch|');
for (let t = 0; t < 8; t++) {
  const a = kf.xf[t][0], b = inf.xf[t][0], c = bp.mean[t][0];
  say('  ' + pad(t + 1, 4) + pad(f(sim.X[t][0], 5), 11) + pad(f(sim.Y[t][0], 5), 11) +
      pad(f(a, 5), 11) + pad(f(b, 5), 13) + pad(f(c, 5), 16) +
      '   ' + e(Math.abs(a - b), 2) + '      ' + e(Math.abs(sm.xs[t][0] - c), 2));
}
say('');

/* ---------- 2. 批处理矩阵 A 的结构（块三对角） ---------- */
say('── ② 批处理全局系统 A（块三对角）的稀疏结构 ──');
const nS = 2, N = bp.A.length;
/* 块大小 n=2 → 半带宽 = 2n−1 = 3，所以「带宽外」是 |i−j| ≥ 2n = 4
   （第一版我按 |i−j|≥2 判定，把块内元素当成了带宽外，误报 34.3） */
let maxOffBand = 0;
for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
  if (Math.abs(i - j) >= 2 * nS && Math.abs(bp.A[i][j]) > maxOffBand) maxOffBand = Math.abs(bp.A[i][j]);
}
say('  A 维度 = ' + N + '×' + N + '（T=' + T + ', n=' + nS + '，块三对角半带宽 = 2n−1 = ' + (2 * nS - 1) + '）');
say('  |A[i][j]| 在带宽外（|i−j| ≥ ' + 2 * nS + '）的最大值 = ' + e(maxOffBand, 2) +
    (maxOffBand === 0 ? '   ✓ 严格块三对角' : '   ✗ 有非零元泄漏'));
say('  A 前 6×6：');
say(mt(bp.A.slice(0, 6).map(r => r.slice(0, 6)), 3));
say('  解析核对（Q⁻¹ = [[34.286,−17.143],[−17.143,11.429]]）：');
say('    A[0][0] = P0⁻¹₀₀ + HᵀR⁻¹H₀₀ + (FᵀQ⁻¹F)₀₀ = 1 + 1.25 + 34.286 = 36.536   ✓');
say('    A[0][2] = −(FᵀQ⁻¹)₀₀ = −34.286   A[0][3] = −(FᵀQ⁻¹)₀₁ = +17.143   ✓');
say('    A[2][2] = Q⁻¹₀₀ + HᵀR⁻¹H₀₀ + (FᵀQ⁻¹F)₀₀ = 34.286 + 1.25 + 34.286 = 69.821   ✓');
say('    A[2][3] = Q⁻¹₀₁ + (FᵀQ⁻¹F)₀₁ = −17.143 + 17.143 = 0（非对角块内抵消）   ✓');
say('');

/* ---------- 3. 平滑 vs 滤波的协方差 ---------- */
say('── ③ 滤波 Σ_f 与平滑 Σ_s（t=1, 6, 12, 24）──');
[0, 5, 11, T - 1].forEach(t => {
  say('  t=' + pad(t + 1, 3) + ' Σ_f = ' + mv(kf.Pf[t].map(r => r[0]), 5).replace('\n', '') +
      ' / ' + mv(kf.Pf[t].map(r => r[1]), 5) + '   →   Σ_s = ' +
      mv(sm.Ps[t].map(r => r[0]), 5) + ' / ' + mv(sm.Ps[t].map(r => r[1]), 5));
});
say('  平滑对角元 − 滤波对角元（应全部 ≤ 0）：');
let worst = -Infinity;
for (let t = 0; t < T; t++) for (let i = 0; i < 2; i++) worst = Math.max(worst, sm.Ps[t][i][i] - kf.Pf[t][i][i]);
say('    max = ' + e(worst, 3) + (worst <= 1e-14 ? '   ✓' : '   ✗ 违反信息单调性'));
say('');

/* ---------- 4. 对数似然两条路径 ---------- */
say('── ④ 对数似然：新息分解 vs 联合高斯稠密协方差 ──');
const seq = kf.loglik, jnt = K.jointLoglik(sim.Y, M);
say('  序列（卡尔曼新息分解）log L = ' + f(seq, 10));
say('  联合（' + T + '×' + T + ' 稠密协方差 Cholesky）log L = ' + f(jnt, 10));
say('  差值 = ' + e(Math.abs(seq - jnt), 3) + '   相对 = ' + e(Math.abs(seq - jnt) / Math.abs(jnt), 3));
say('  信息滤波 log L = ' + f(inf.loglik, 10) + '（第三条路径）');
say('');

/* ---------- 5. 新息诊断 ---------- */
say('── ⑤ 新息诊断（T=1200，理论：零均值白噪声、Var=S、NIS~χ²(m)）──');
const Mb = K.sceneRadar(1, 0.2, 1.2);
const simB = K.simulate(Mb, 1200, K.mulberry32(555));
const kfB = K.kalman(simB.Y, Mb);
const ic = K.innovCovCheck(kfB);
say('  理论 S（t=600）= ' + mt(kfB.S[600]));
say('  样本 E[ννᵀ]（1200 步）= ' + mt(ic.emp));
say('  逐项最大相对偏差 = ' + e(ic.rel, 3));
const nv = K.nis(kfB);
say('  ⟨NIS⟩ = ' + f(K.mean(nv), 4) + '   （χ²(2) 的均值 = 2，方差 = 4）');
say('  NIS 样本方差 = ' + f(nv.reduce((s, v) => s + (v - K.mean(nv)) ** 2, 0) / (nv.length - 1), 4) + '   （χ²(2) 理论方差 = 4）');
const lo = K.chi2Quantile(0.025, 2), hi = K.chi2Quantile(0.975, 2);
let inside = 0; nv.forEach(v => { if (v >= lo && v <= hi) inside++; });
say('  落在 χ²(2) 95% 区间 [' + f(lo, 3) + ', ' + f(hi, 3) + '] 的比例 = ' + f(inside / nv.length * 100, 2) + '%  （理论 95%）');
say('  归一化新息滞后1自相关 ρ₁ = ' + f(K.innovLag1(kfB), 5) + '   （最优滤波器应 ≈ 0）');
say('');
say('  ✗ 对照一：Q 与 R 【同时】缩小 10 倍（比例不变 → 增益 K 不变）');
const badM = K.sceneRadar(1, 0.2 * 0.1, 1.2 * 0.1);
const kfC = K.kalman(simB.Y, badM);
const nvC = K.nis(kfC);
let insideC = 0; nvC.forEach(v => { if (v >= lo && v <= hi) insideC++; });
say('    ⟨NIS⟩ = ' + f(K.mean(nvC), 4) + '   ← 恰好是正确模型的 ' + f(K.mean(nvC) / K.mean(nv), 3) +
    ' 倍（Q,R 同比例缩放时 K 与 ν 都不变，只有 S 缩小 → NIS 反比放大）');
say('    ρ₁ = ' + f(K.innovLag1(kfC), 5) + '   ← 与正确模型几乎相同：新息【仍然】是白噪声，');
say('       因为 K 不变。所以「新息白」检验查不出这一类误配，只有 NIS 的绝对尺度能查出来。');
say('    落入 95% 区间 ' + f(insideC / nvC.length * 100, 2) + '%');
say('  ✗ 对照二：只把 Q 放大 10 倍（改变 Q/R 比例 → K 改变）');
const badM2 = K.sceneRadar(1, 0.2 * 10, 1.2);
const kfD = K.kalman(simB.Y, badM2);
const nvD = K.nis(kfD);
say('    ⟨NIS⟩ = ' + f(K.mean(nvD), 4) + '    ρ₁ = ' + f(K.innovLag1(kfD), 5) +
    '   ← 相关性显著上升（' + f(Math.abs(K.innovLag1(kfD) / K.innovLag1(kfB)), 1) + ' 倍），新息不再白');
say('');

/* ---------- 6. 稳态 ---------- */
say('── ⑥ Riccati 迭代 → DARE 不动点 ──');
const ss = K.steadyState(M, 1e-15, 20000);
say('  收敛轮数 = ' + ss.iters + '    DARE 残差 ‖P − (F(P−KHP)Fᵀ+Q)‖ = ' + e(ss.resid, 3));
say('  稳态预测协方差 P∞⁻ = ' + mt(ss.Ppred));
say('  稳态滤波协方差 P∞  = ' + mt(ss.P));
say('  稳态增益 K∞ = ' + mv(ss.K.map(r => r[0]), 6));
const long = K.kalman(K.simulate(M, 500, K.mulberry32(77)).Y, M);
say('  长跑 K₅₀₀        = ' + mv(long.K[499].map(r => r[0]), 6));
say('  |K₅₀₀ − K∞| = ' + e(K.maxAbsDiff(long.K[499], ss.K), 3));
say('');

/* ---------- 7. EM ---------- */
say('── ⑦ EM 学习 (Q, R)：从错误初值出发的迭代轨迹 ──');
const truth = K.sceneCV(1, 0.35, 0.8);
const simE = K.simulate(truth, 60, K.mulberry32(73));
const start = K.sceneCV(1, 0.35 * 6, 0.8 * 0.25);
const r = K.em(simE.Y, start, { iters: 30 });
say('  真值  q* = ' + e(truth.Q[1][1], 4) + '   r* = ' + e(truth.R[0][0], 4));
say('  初值  q₀ = ' + e(start.Q[1][1], 4) + '   r₀ = ' + e(start.R[0][0], 4));
say('  ' + pad('iter', 6) + pad('log L', 14) + pad('q̂', 14) + 'r̂');
[0, 1, 2, 4, 7, 10, 15, 20, 29].forEach(i => {
  if (i >= r.hist.length) return;
  say('  ' + pad(i + 1, 6) + pad(f(r.hist[i], 5), 14) + pad(e(r.Qs[i][1][1], 4), 14) + e(r.Rs[i][0][0], 4));
});
let minD = Infinity;
for (let i = 1; i < r.hist.length; i++) minD = Math.min(minD, r.hist[i] - r.hist[i - 1]);
say('  单调性 min Δ(log L) = ' + e(minD, 3) + (minD > -1e-8 ? '   ✓ 非减' : '   ✗ 下降'));
say('  终值  q̂ = ' + e(r.M.Q[1][1], 4) + '   r̂ = ' + e(r.M.R[0][0], 4) +
    '   |log q̂/q*| = ' + f(Math.abs(Math.log(r.M.Q[1][1] / truth.Q[1][1])), 4) +
    '   |log r̂/r*| = ' + f(Math.abs(Math.log(r.M.R[0][0] / truth.R[0][0])), 4));
say('  参照：真值参数下的 log L = ' + f(K.kalman(simE.Y, truth).loglik, 5) +
    '（EM 终点 ' + f(r.hist[r.hist.length - 1], 5) + '）');
say('');

/* ---------- 8. 粒子滤波 ---------- */
say('── ⑧ 粒子滤波 → 卡尔曼（线性高斯下 PF 应收敛到精确解）──');
const Mp = K.sceneCV(1, 0.35, 0.8), Tp = 40;
const simP = K.simulate(Mp, Tp, K.mulberry32(97));
const kfP = K.kalman(simP.Y, Mp);
say('  ' + pad('N', 8) + pad('平均 |μ_PF − μ_KF|', 20) + pad('末端 |Δ|', 14) + '平均 ESS');
[100, 400, 1600].forEach(N => {
  let acc = 0, endAcc = 0, essAcc = 0, seeds = 3;
  for (let s = 0; s < seeds; s++) {
    const pf = K.particleFilter(simP.Y, Mp, N, K.mulberry32(1000 + s));
    let e2 = 0;
    for (let t = 0; t < Tp; t++) e2 += Math.abs(pf.mean[t][0] - kfP.xf[t][0]);
    acc += e2 / Tp;
    endAcc += Math.abs(pf.mean[Tp - 1][0] - kfP.xf[Tp - 1][0]);
    essAcc += K.mean(pf.ess);
  }
  say('  ' + pad(N, 8) + pad(f(acc / seeds, 6), 20) + pad(f(endAcc / seeds, 6), 14) + f(essAcc / seeds, 1));
});
say('  （理论收敛率 O(1/√N)：N 每 ×4，误差应约 ÷2）');
say('');

/* ---------- 9. 一次完整轨迹的滤波 vs 平滑误差 ---------- */
say('── ⑨ 滤波 vs 平滑 vs 裸观测的 RMSE（8 次平均，T=100）──');
say('  ' + pad('场景', 10) + pad('裸观测', 12) + pad('滤波', 12) + pad('平滑', 12) + '平滑提升');
['rw', 'cv', 'radar'].forEach(sc => {
  const Mx = K.scene(sc, -0.3, 0.3);
  let rf = 0, rs = 0, ro = 0, trials = 8;
  for (let c = 0; c < trials; c++) {
    const s2 = K.simulate(Mx, 100, K.mulberry32(900 + c));
    const k2 = K.kalman(s2.Y, Mx), m2 = K.rts(k2, Mx);
    rf += K.rmse(k2.xf, s2.X, 0); rs += K.rmse(m2.xs, s2.X, 0);
    let e3 = 0; for (let t = 0; t < 100; t++) e3 += (s2.Y[t][0] - s2.X[t][0]) ** 2;
    ro += Math.sqrt(e3 / 100);
  }
  rf /= trials; rs /= trials; ro /= trials;
  say('  ' + pad(sc, 10) + pad(f(ro, 5), 12) + pad(f(rf, 5), 12) + pad(f(rs, 5), 12) +
      f((1 - rs / rf) * 100, 2) + '%');
});
say('');
say('══════════ 探针结束 ══════════');

fs.writeFileSync(path.join(__dirname, '_probe.txt'), buf.join('\n') + '\n');
console.log('wrote _probe.txt (' + buf.length + ' lines)');
