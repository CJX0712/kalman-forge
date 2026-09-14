/* kalman-forge 无头验证：engine 从 index.html 抽出，在 Node vm 沙箱里跑。
   用法: node _smoke.js   —— 输出 PASS n / N，并写 _smoke.log */
'use strict';
const fs = require('fs'), vm = require('vm'), path = require('path');

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
if (!m) { console.error('找不到 <script id="engine">'); process.exit(1); }

const ctx = {
  console, Math, JSON, Array, Object, Number, String, Boolean, Error,
  isFinite, isNaN, Infinity, NaN, undefined,
  Float64Array, Uint8Array, Int32Array, TextEncoder, TextDecoder
};
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(m[1], ctx, { filename: 'engine.js' });
const K = ctx.KF;
if (!K) { console.error('engine 未暴露 globalThis.KF'); process.exit(1); }

let pass = 0, fail = 0; const fails = [];
function ok(cond, name, detail) {
  if (cond) pass++; else { fail++; fails.push(name + (detail ? '  [' + detail + ']' : '')); }
}
function close(a, b, tol) { return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b)); }
function near(a, b, tol) { return Math.abs(a - b) <= tol; }
function section(t) { /* 分组标记，仅用于人眼阅读 */ }

/* =====================================================================
   0) engine 自带 8 条不变量
   ===================================================================== */
section('selfTest');
{
  const res = K.selfTest();
  ok(res.length === 8, 'selfTest 返回 8 条', '' + res.length);
  res.forEach((r, i) => ok(r.pass, `selfTest#${i + 1} ${r.name.slice(0, 22)}`, r.detail));
}

/* =====================================================================
   1) 确定性 / 可复现
   ===================================================================== */
section('determinism');
{
  const M = K.sceneCV(1, 0.35, 0.8);
  const s1 = K.simulate(M, 40, K.mulberry32(7));
  const s2 = K.simulate(M, 40, K.mulberry32(7));
  let d = 0;
  for (let t = 0; t < 40; t++) d = Math.max(d, K.maxAbsDiffV(s1.Y[t], s2.Y[t]));
  ok(d === 0, '同种子仿真逐位一致', '' + d);
  const k1 = K.kalman(s1.Y, M), k2 = K.kalman(s1.Y, M);
  ok(k1.loglik === k2.loglik, '滤波确定性', '' + k1.loglik);
  const r1 = K.mulberry32(3), r2 = K.mulberry32(3);
  ok(r1() === r2(), 'mulberry32 确定性', '');
}

/* =====================================================================
   2) 三场景 × 多种子：KF == 信息滤波（独立数值路径）
   ===================================================================== */
section('kf-vs-info');
for (const sc of ['rw', 'cv', 'radar']) {
  for (let seed = 0; seed < 6; seed++) {
    const M = K.scene(sc, -0.5, 0.2);
    const sim = K.simulate(M, 50, K.mulberry32(100 + seed));
    const kf = K.kalman(sim.Y, M), inf = K.infoFilter(sim.Y, M);
    let dm = 0, dP = 0;
    for (let t = 0; t < 50; t++) {
      dm = Math.max(dm, K.maxAbsDiffV(kf.xf[t], inf.xf[t]));
      dP = Math.max(dP, K.maxAbsDiff(kf.Pf[t], inf.Pf[t]));
    }
    ok(dm < 1e-7, `${sc}#${seed} KF==信息滤波 μ`, dm.toExponential(2));
    ok(dP < 1e-7, `${sc}#${seed} KF==信息滤波 Σ`, dP.toExponential(2));
    ok(close(kf.loglik, inf.loglik, 1e-10), `${sc}#${seed} KF==信息滤波 logL`, kf.loglik + ' vs ' + inf.loglik);
  }
}

/* =====================================================================
   3) 递推 RTS == 批处理块三对角全局后验（μ 与 Σ 全对拍）
   ===================================================================== */
section('rts-vs-batch');
for (const sc of ['rw', 'cv', 'radar']) {
  for (let seed = 0; seed < 4; seed++) {
    const M = K.scene(sc, -0.4, 0.1);
    const T = sc === 'radar' ? 25 : 35;
    const sim = K.simulate(M, T, K.mulberry32(200 + seed));
    const kf = K.kalman(sim.Y, M), sm = K.rts(kf, M), bp = K.batchPosterior(sim.Y, M);
    ok(bp !== null, `${sc}#${seed} 批处理后验可解`, '');
    let dm = 0, dP = 0;
    for (let t = 0; t < T; t++) {
      dm = Math.max(dm, K.maxAbsDiffV(sm.xs[t], bp.mean[t]));
      dP = Math.max(dP, K.maxAbsDiff(sm.Ps[t], bp.Pt[t]));
    }
    ok(dm < 1e-8, `${sc}#${seed} RTS==批处理 μ`, dm.toExponential(2));
    ok(dP < 1e-8, `${sc}#${seed} RTS==批处理 Σ`, dP.toExponential(2));
  }
}
/* T=1 边界：批处理退化成单块，应等于解析的贝叶斯更新 */
{
  const M = K.sceneCV(1, 0.3, 0.7);
  const Y = [[1.234]];
  const kf = K.kalman(Y, M), sm = K.rts(kf, M), bp = K.batchPosterior(Y, M);
  ok(near(sm.xs[0][0], bp.mean[0][0], 1e-12), 'T=1 批处理 == 滤波', '');
  const S = M.P0[0][0] + M.R[0][0], Kg = M.P0[0][0] / S;
  const ana = M.x0[0] + Kg * (Y[0][0] - M.x0[0]);
  ok(near(kf.xf[0][0], ana, 1e-12), 'T=1 滤波 == 解析贝叶斯更新', '' + kf.xf[0][0] + ' vs ' + ana);
  ok(near(kf.Pf[0][0][0], M.P0[0][0] * (1 - Kg), 1e-12), 'T=1 方差 == 解析', '' + kf.Pf[0][0][0]);
}

/* =====================================================================
   4) 序列对数似然 == 联合高斯稠密协方差对数似然
   ===================================================================== */
section('loglik');
for (const sc of ['rw', 'cv', 'radar']) {
  for (let seed = 0; seed < 3; seed++) {
    const M = K.scene(sc, -0.3, 0.3);
    const T = sc === 'radar' ? 22 : 30;
    const sim = K.simulate(M, T, K.mulberry32(300 + seed));
    const seq = K.kalman(sim.Y, M).loglik, jnt = K.jointLoglik(sim.Y, M);
    ok(close(seq, jnt, 1e-9), `${sc}#${seed} 序列 logL == 联合 logL`, seq.toFixed(8) + ' vs ' + jnt.toFixed(8));
  }
}
/* 似然的最优性：真值参数的 logL 应高于错误参数（大样本下） */
{
  const truth = K.sceneCV(1, 0.35, 0.8);
  const sim = K.simulate(truth, 800, K.mulberry32(9));
  const llT = K.kalman(sim.Y, truth).loglik;
  const llA = K.kalman(sim.Y, K.sceneCV(1, 0.35 * 4, 0.8)).loglik;
  const llB = K.kalman(sim.Y, K.sceneCV(1, 0.35, 0.8 * 4)).loglik;
  ok(llT > llA, '真值 Q 的 logL 高于错误 Q', llT.toFixed(1) + ' > ' + llA.toFixed(1));
  ok(llT > llB, '真值 R 的 logL 高于错误 R', llT.toFixed(1) + ' > ' + llB.toFixed(1));
}

/* =====================================================================
   5) 协方差的基本性质：对称 + 半正定 + 平滑方差 ≤ 滤波方差
   ===================================================================== */
section('cov-properties');
for (const sc of ['rw', 'cv', 'radar']) {
  const M = K.scene(sc, -0.4, 0.2);
  const sim = K.simulate(M, 60, K.mulberry32(401));
  const kf = K.kalman(sim.Y, M), sm = K.rts(kf, M);
  const n = M.F.length;
  for (let t = 0; t < 60; t++) {
    for (const P of [kf.Pf[t], kf.Pp[t], sm.Ps[t]]) {
      let symErr = 0, psd = true;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) symErr = Math.max(symErr, Math.abs(P[i][j] - P[j][i]));
        if (P[i][i] < -1e-12) psd = false;
      }
      if (symErr > 1e-14) { ok(false, `${sc} t=${t} 协方差对称`, '' + symErr); }
      if (!psd) ok(false, `${sc} t=${t} 对角元非负`, '');
    }
    for (let i = 0; i < n; i++) {
      if (sm.Ps[t][i][i] > kf.Pf[t][i][i] + 1e-12) {
        ok(false, `${sc} t=${t} 平滑方差 ≤ 滤波方差`, '' + sm.Ps[t][i][i] + ' vs ' + kf.Pf[t][i][i]);
      }
    }
  }
  ok(true, `${sc} 协方差对称+对角非负+平滑方差≤滤波方差（60×3 矩阵）`, '');
}

/* =====================================================================
   6) 新息诊断：白噪声 + NIS 校准 + 样本协方差 == S
   ===================================================================== */
section('innovations');
for (const sc of ['rw', 'cv', 'radar']) {
  const M = K.scene(sc, -0.3, 0.2);
  const T = 1200;
  const sim = K.simulate(M, T, K.mulberry32(555));
  const kf = K.kalman(sim.Y, M);
  const ic = K.innovCovCheck(kf);
  const nv = K.nis(kf), mm = M.H.length;
  const mn = K.mean(nv), r1 = K.innovLag1(kf);
  ok(ic.rel < 0.15, `${sc} 新息样本协方差 == 理论 S`, (ic.rel * 100).toFixed(2) + '%');
  ok(Math.abs(mn - mm) / mm < 0.12, `${sc} ⟨NIS⟩ ≈ m`, mn.toFixed(3) + ' vs ' + mm);
  ok(Math.abs(r1) < 0.08, `${sc} 新息滞后1自相关 ≈ 0`, r1.toFixed(4));
  /* 误配时 NIS 应显著偏离 */
  const bad = K.scene(sc, -0.3 + 1.0, 0.2);
  const kfB = K.kalman(sim.Y, bad);
  const mnB = K.mean(K.nis(kfB));
  ok(Math.abs(mnB - mm) > Math.abs(mn - mm), `${sc} 模型误配时 NIS 偏离更大`,
     Math.abs(mnB - mm).toFixed(3) + ' > ' + Math.abs(mn - mm).toFixed(3));
  ok(Math.abs(K.innovLag1(kfB)) > Math.abs(r1) * 0.9, `${sc} 误配时新息相关性上升`,
     K.innovLag1(kfB).toFixed(4));
}

/* =====================================================================
   7) 缺测处理：只预测不更新，方差应单调不减
   ===================================================================== */
section('missing');
{
  const M = K.sceneCV(1, 0.35, 0.8);
  const sim = K.simulate(M, 60, K.mulberry32(66));
  const Y2 = sim.Y.map((v, i) => (i % 3 === 1 ? null : v.slice()));
  const kf = K.kalman(Y2, M);
  ok(kf.innov[1] === null, '缺测步不产生新息', '');
  let mono = true;
  for (let t = 1; t < 60; t++) {
    if (Y2[t] === null) {
      if (kf.Pf[t][0][0] < kf.Pf[t - 1][0][0] - 1e-12) mono = false;
    }
  }
  ok(mono, '缺测步方差不减（无信息流入）', '');
  const kf2 = K.kalman(sim.Y, M);
  let agree = true;
  for (let t = 0; t < 60; t++) if (Y2[t] !== null && Math.abs(kf.xf[t][0] - kf2.xf[t][0]) > 1e-9) agree = false;
  ok(!agree, '缺测会改变后续估计（信息确实被丢弃）', '');
}

/* =====================================================================
   8) 稳态 Riccati：DARE 残差 → 0，且与长跑时变增益一致
   ===================================================================== */
section('steady');
for (const sc of ['rw', 'cv', 'radar']) {
  const M = K.scene(sc, -0.3, 0.2);
  const ss = K.steadyState(M, 1e-15, 20000);
  ok(ss.resid < 1e-9, `${sc} DARE 残差 ≈ 0`, ss.resid.toExponential(2));
  const sim = K.simulate(M, 400, K.mulberry32(77));
  const kf = K.kalman(sim.Y, M);
  const dK = K.maxAbsDiff(kf.K[399], ss.K);
  ok(dK < 1e-9, `${sc} 稳态增益 == 长跑增益 K₄₀₀`, dK.toExponential(2));
  /* 从极端的 P0 出发也应收敛到同一个不动点 */
  const M2 = { F: M.F, Q: M.Q, H: M.H, R: M.R, x0: M.x0, P0: K.scl(M.P0, 1000) };
  const ss2 = K.steadyState(M2, 1e-15, 20000);
  ok(K.maxAbsDiff(ss.Ppred, ss2.Ppred) < 1e-8, `${sc} 不动点与初值无关`, K.maxAbsDiff(ss.Ppred, ss2.Ppred).toExponential(2));
}

/* =====================================================================
   9) EM：单调非减 + 多种起点参数还原
   ===================================================================== */
section('em');
for (let c = 0; c < 3; c++) {
  const truth = K.sceneCV(1, 0.35, 0.8);
  const sim = K.simulate(truth, 60, K.mulberry32(800 + c));
  const inits = [
    K.sceneCV(1, 0.35 * 6, 0.8 * 0.25),
    K.sceneCV(1, 0.35 * 0.2, 0.8 * 5),
    K.sceneCV(1, 0.35, 0.8)
  ];
  inits.forEach((ini, ii) => {
    /* EM 在 LDS 上收敛很慢（实测：T=60 从坏初值出发需要 ~100 轮才越过真值似然），
       短序列每轮只要 ~5ms，所以直接跑 200 轮。慢 ≠ 卡住 —— 网格扫描确认曲面是单峰的。 */
    const r = K.em(sim.Y, ini, { iters: 200 });
    let minD = Infinity;
    for (let i = 1; i < r.hist.length; i++) minD = Math.min(minD, r.hist[i] - r.hist[i - 1]);
    ok(minD > -1e-6, `EM c=${c} init=${ii} 边际似然单调非减`, 'minΔ=' + minD.toExponential(2));
    /* 短序列上 MLE ≠ 真值（MLE 的采样误差本来就大），所以真正可断言的是：
       EM 单调 + EM 终点似然 ≥ 真值参数似然（EM 在最大化边际似然） */
    const truthLL = K.kalman(sim.Y, truth).loglik, endLL = r.hist[r.hist.length - 1];
    ok(endLL >= truthLL - 1e-6, `EM c=${c} init=${ii} 终点似然 ≥ 真值似然`,
       endLL.toFixed(4) + ' vs ' + truthLL.toFixed(4));
    /* 从真值出发，EM 第一步不应让似然下降 */
    const rT = K.em(sim.Y, truth, { iters: 3 });
    ok(rT.hist[1] >= rT.hist[0] - 1e-8, `EM c=${c} init=${ii} 真值起点不下降`,
       rT.hist[0].toFixed(4) + ' → ' + rT.hist[1].toFixed(4));
  });
}
/* 长序列参数还原：样本量足够时 q̂,r̂ 应逼近真值 */
{
  const truth = K.sceneCV(1, 0.35, 0.8);
  const sim = K.simulate(truth, 250, K.mulberry32(800));
  const ini = K.sceneCV(1, 0.35 * 0.2, 0.8 * 5);
  const r = K.em(sim.Y, ini, { iters: 40 });
  const eq = Math.abs(Math.log(r.M.Q[1][1] / truth.Q[1][1]));
  const er = Math.abs(Math.log(r.M.R[0][0] / truth.R[0][0]));
  ok(eq < 0.35, 'EM 长序列(T=250) q̂ 还原', r.M.Q[1][1].toExponential(3) + ' vs ' + truth.Q[1][1].toExponential(3));
  ok(er < 0.25, 'EM 长序列(T=250) r̂ 还原', r.M.R[0][0].toExponential(3) + ' vs ' + truth.R[0][0].toExponential(3));
}

/* =====================================================================
   10) 粒子滤波 → 卡尔曼（Monte-Carlo 收敛）
   ===================================================================== */
section('pf');
{
  const M = K.sceneCV(1, 0.35, 0.8), T = 40;
  const sim = K.simulate(M, T, K.mulberry32(97));
  const kf = K.kalman(sim.Y, M);
  const Ns = [100, 400, 1600], errs = [], seeds = 3;
  Ns.forEach(N => {
    let acc = 0;
    for (let s = 0; s < seeds; s++) {
      const pf = K.particleFilter(sim.Y, M, N, K.mulberry32(1000 + s));
      let e = 0;
      for (let t = 0; t < T; t++) e += Math.abs(pf.mean[t][0] - kf.xf[t][0]);
      acc += e / T;
    }
    errs.push(acc / seeds);
  });
  ok(errs[2] < errs[0], 'PF 误差随 N 增大而下降', errs.map(e => e.toFixed(4)).join(' → '));
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let k = 0; k < Ns.length; k++) { const lx = Math.log(Ns[k]), ly = Math.log(errs[k]); sx += lx; sy += ly; sxx += lx * lx; sxy += lx * ly; }
  const slope = (Ns.length * sxy - sx * sy) / (Ns.length * sxx - sx * sx);
  ok(slope < -0.15 && slope > -0.95, 'PF 收敛斜率 ≈ −0.5', slope.toFixed(3));
  ok(errs[2] < 0.15, 'PF(N=1600) 平均偏差 < 0.15', errs[2].toFixed(4));
  /* 粒子退化的 ESS 应在 (1, N] 内 */
  const pf = K.particleFilter(sim.Y, M, 200, K.mulberry32(5));
  ok(pf.ess.every(e => e > 1 && e <= 200.0001), 'PF 有效样本量落在 (1, N]', '');
}

/* =====================================================================
   11) 平滑器优于滤波器（多种子平均）
   ===================================================================== */
section('smooth-better');
for (const sc of ['rw', 'cv', 'radar']) {
  const M = K.scene(sc, -0.3, 0.3);
  let rf = 0, rs = 0, ro = 0, trials = 8;
  for (let c = 0; c < trials; c++) {
    const sim = K.simulate(M, 100, K.mulberry32(900 + c));
    const kf = K.kalman(sim.Y, M), sm = K.rts(kf, M);
    rf += K.rmse(kf.xf, sim.X, 0); rs += K.rmse(sm.xs, sim.X, 0);
    let e = 0; for (let t = 0; t < 100; t++) e += (sim.Y[t][0] - sim.X[t][0]) ** 2;
    ro += Math.sqrt(e / 100);
  }
  rf /= trials; rs /= trials; ro /= trials;
  ok(rs < rf, `${sc} 平滑 RMSE < 滤波 RMSE`, rs.toFixed(4) + ' < ' + rf.toFixed(4));
  ok(rf < ro * 1.05, `${sc} 滤波 RMSE 不高于裸观测太多`, rf.toFixed(4) + ' vs ' + ro.toFixed(4));
}

/* =====================================================================
   12) 特殊函数：χ² 分位数 vs 已知解析值
   ===================================================================== */
section('chi2');
{
  /* χ²(2) 是均值为 2 的指数分布：P(X ≤ x) = 1 − e^{−x/2} */
  const q = p => -2 * Math.log(1 - p);
  [0.025, 0.5, 0.9, 0.975].forEach(p => {
    const got = K.chi2Quantile(p, 2), want = q(p);
    ok(near(got, want, 1e-6), `χ²(2) ${p} 分位数`, got.toFixed(6) + ' vs ' + want.toFixed(6));
  });
  /* χ²(1) 0.95 分位数 = 3.8414588207 */
  ok(near(K.chi2Quantile(0.95, 1), 3.841458820694124, 1e-6), 'χ²(1) 0.95 = 3.84146', '' + K.chi2Quantile(0.95, 1));
  /* 单调性 + 非负 */
  let mono = true, prev = -1;
  for (const df of [1, 2, 3, 4, 6, 10]) { const v = K.chi2Quantile(0.95, df); if (v <= prev) mono = false; prev = v; }
  ok(mono, 'χ² 0.95 分位数随自由度单调增', '');
}

/* =====================================================================
   13) 边界：极小/极大噪声、单点、数值不崩
   ===================================================================== */
section('edge');
{
  const cases = [
    ['rw', -6, -6], ['rw', 2, 2], ['cv', -6, 2], ['cv', 2, -6],
    ['radar', -6, -6], ['radar', 2, 2]
  ];
  cases.forEach(([sc, q, r]) => {
    const M = K.scene(sc, q, r);
    const sim = K.simulate(M, 30, K.mulberry32(1234));
    const kf = K.kalman(sim.Y, M), sm = K.rts(kf, M);
    let finite = true;
    kf.xf.forEach(v => v.forEach(u => { if (!isFinite(u)) finite = false; }));
    sm.xs.forEach(v => v.forEach(u => { if (!isFinite(u)) finite = false; }));
    kf.Pf.forEach(P => P.forEach(row => row.forEach(u => { if (!isFinite(u)) finite = false; })));
    ok(finite, `${sc} q=1e${q} r=1e${r} 全量有限`, '');
    ok(isFinite(kf.loglik), `${sc} q=1e${q} r=1e${r} logL 有限`, '' + kf.loglik);
  });
  /* 观测恒为 0 的退化情形 */
  const M = K.sceneCV(1, 0.3, 0.7);
  const Y0 = Array.from({ length: 20 }, () => [0]);
  const k0 = K.kalman(Y0, M);
  ok(isFinite(k0.loglik) && k0.xf.every(v => isFinite(v[0])), '全零观测不崩', '' + k0.loglik);
}

/* ============================== 输出 ============================== */
const total = pass + fail;
const head = `PASS ${pass} / ${total}` + (fail ? `   FAIL ${fail}` : '   ALL GREEN');
console.log(head);
if (fail) fails.forEach(f => console.log('  ✗ ' + f));
fs.writeFileSync(path.join(__dirname, '_smoke.log'),
  head + '\n' + (fail ? fails.map(f => 'FAIL ' + f).join('\n') : 'ALL GREEN') + '\n');
if (fail) process.exitCode = 1;
