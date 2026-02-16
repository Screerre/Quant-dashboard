import { useState, useMemo, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════════════
// MATHEMATICAL ENGINE
// ═══════════════════════════════════════════════════════════════════════

const normCDF = (x) => {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const t = 1.0 / (1.0 + p * Math.abs(x));
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x / 2);
  return 0.5 * (1.0 + sign * y);
};
const normPDF = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

const blackScholes = (S, K, T, r, sigma, type = "call") => {
  if (T <= 0.0001) {
    if (type === "call") return { price: Math.max(S - K, 0), delta: S > K ? 1 : 0, gamma: 0, theta: 0, vega: 0, rho: 0 };
    return { price: Math.max(K - S, 0), delta: S < K ? -1 : 0, gamma: 0, theta: 0, vega: 0, rho: 0 };
  }
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  if (type === "call") {
    const price = S * normCDF(d1) - K * Math.exp(-r * T) * normCDF(d2);
    const delta = normCDF(d1);
    const gamma = normPDF(d1) / (S * sigma * Math.sqrt(T));
    const theta = (-(S * normPDF(d1) * sigma) / (2 * Math.sqrt(T)) - r * K * Math.exp(-r * T) * normCDF(d2)) / 365;
    const vega = S * normPDF(d1) * Math.sqrt(T) / 100;
    const rho = K * T * Math.exp(-r * T) * normCDF(d2) / 100;
    return { price, delta, gamma, theta, vega, rho, d1, d2 };
  } else {
    const price = K * Math.exp(-r * T) * normCDF(-d2) - S * normCDF(-d1);
    const delta = normCDF(d1) - 1;
    const gamma = normPDF(d1) / (S * sigma * Math.sqrt(T));
    const theta = (-(S * normPDF(d1) * sigma) / (2 * Math.sqrt(T)) + r * K * Math.exp(-r * T) * normCDF(-d2)) / 365;
    const vega = S * normPDF(d1) * Math.sqrt(T) / 100;
    const rho = -K * T * Math.exp(-r * T) * normCDF(-d2) / 100;
    return { price, delta, gamma, theta, vega, rho, d1, d2 };
  }
};

const boxMuller = () => {
  let u1 = Math.random(), u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
};

const hestonMC = (S0, K, T, r, v0, kappa, theta_h, xi, rho_h, nPaths, nSteps, optType = "call") => {
  const dt = T / nSteps;
  const sqrtDt = Math.sqrt(dt);
  const payoffs = [];
  const paths = [];
  const volPaths = [];
  for (let i = 0; i < nPaths; i++) {
    let S = S0, v = v0;
    const path = [S];
    const vPath = [Math.sqrt(v) * 100];
    for (let j = 0; j < nSteps; j++) {
      const z1 = boxMuller();
      const z2 = rho_h * z1 + Math.sqrt(1 - rho_h * rho_h) * boxMuller();
      v = Math.max(v + kappa * (theta_h - v) * dt + xi * Math.sqrt(Math.max(v, 0)) * sqrtDt * z2, 0.0001);
      S = S * Math.exp((r - 0.5 * v) * dt + Math.sqrt(Math.max(v, 0)) * sqrtDt * z1);
      path.push(S);
      vPath.push(Math.sqrt(v) * 100);
    }
    paths.push(path);
    volPaths.push(vPath);
    if (optType === "call") payoffs.push(Math.max(S - K, 0));
    else payoffs.push(Math.max(K - S, 0));
  }
  const discPayoffs = payoffs.map(p => p * Math.exp(-r * T));
  const price = discPayoffs.reduce((a, b) => a + b) / nPaths;
  const probITM = payoffs.filter(p => p > 0).length / nPaths;
  const terminals = paths.map(p => p[p.length - 1]);
  const sorted = [...terminals].sort((a, b) => a - b);
  return { price, probITM, paths: paths.slice(0, 60), volPaths: volPaths.slice(0, 60), terminals, sorted, payoffs,
    pct: (p) => sorted[Math.floor(p * sorted.length)] };
};

const computeRiskMetrics = (pnlArray, confidence = 0.95) => {
  const sorted = [...pnlArray].sort((a, b) => a - b);
  const idx = Math.floor((1 - confidence) * sorted.length);
  const VaR = -sorted[idx];
  const tailLosses = sorted.slice(0, idx);
  const CVaR = tailLosses.length > 0 ? -tailLosses.reduce((a, b) => a + b) / tailLosses.length : VaR;
  const maxLoss = -sorted[0];
  const maxGain = sorted[sorted.length - 1];
  const mean = pnlArray.reduce((a, b) => a + b) / pnlArray.length;
  const std = Math.sqrt(pnlArray.reduce((a, b) => a + (b - mean) ** 2, 0) / pnlArray.length);
  const skew = std > 0 ? pnlArray.reduce((a, b) => a + ((b - mean) / std) ** 3, 0) / pnlArray.length : 0;
  const kurt = std > 0 ? pnlArray.reduce((a, b) => a + ((b - mean) / std) ** 4, 0) / pnlArray.length - 3 : 0;
  return { VaR, CVaR, maxLoss, maxGain, mean, std, skew, kurt, sorted };
};

const greeksSurface = (S0, K, T, r, metric = "delta", optType = "call") => {
  const spots = [], vols = [], surface = [];
  for (let s = S0 * 0.85; s <= S0 * 1.15; s += S0 * 0.02) spots.push(Math.round(s));
  for (let v = 10; v <= 40; v += 2) vols.push(v);
  for (let vi = 0; vi < vols.length; vi++) {
    const row = [];
    for (let si = 0; si < spots.length; si++) {
      const g = blackScholes(spots[si], K, T, r, vols[vi] / 100, optType);
      row.push(g[metric]);
    }
    surface.push(row);
  }
  return { spots, vols, surface };
};

// ═══════════════════════════════════════════════════════════════════════
// UI COMPONENTS
// ═══════════════════════════════════════════════════════════════════════

const PRESETS = [
  { name: "Or (Gold)", symbol: "XAU", spot: 5050, strike: 5150, vol: 21.34, rate: 4.5, maturity: 5, type: "call" },
  { name: "S&P 500", symbol: "SPX", spot: 5900, strike: 6000, vol: 16.5, rate: 4.5, maturity: 3, type: "call" },
  { name: "EUR/USD", symbol: "EUR/USD", spot: 1.085, strike: 1.10, vol: 8.2, rate: 3.5, maturity: 6, type: "call" },
  { name: "Pétrole (WTI)", symbol: "WTI", spot: 72, strike: 75, vol: 32, rate: 4.5, maturity: 4, type: "call" },
  { name: "Bitcoin", symbol: "BTC", spot: 97000, strike: 100000, vol: 55, rate: 4.5, maturity: 3, type: "call" },
  { name: "Tesla", symbol: "TSLA", spot: 340, strike: 360, vol: 52, rate: 4.5, maturity: 2, type: "call" },
  { name: "Apple", symbol: "AAPL", spot: 230, strike: 240, vol: 22, rate: 4.5, maturity: 3, type: "call" },
  { name: "Put Or", symbol: "XAU Put", spot: 5050, strike: 4950, vol: 21.34, rate: 4.5, maturity: 5, type: "put" },
];

const Panel = ({ title, number, children, accent = "#B49B50" }) => (
  <div style={{
    background: "linear-gradient(135deg, rgba(12,12,18,0.95), rgba(18,18,28,0.9))",
    border: `1px solid ${accent}22`, borderRadius: 10, padding: "18px 20px", marginBottom: 14
  }}>
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
      <div style={{
        width: 26, height: 26, borderRadius: "50%", background: `${accent}22`,
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 11, fontWeight: 700, color: accent, fontFamily: "monospace"
      }}>{number}</div>
      <h2 style={{ fontSize: 13, fontWeight: 600, color: accent, margin: 0, letterSpacing: 1.5, textTransform: "uppercase" }}>{title}</h2>
    </div>
    {children}
  </div>
);

const Metric = ({ label, value, sub, color = "#D4B96A", small = false }) => (
  <div style={{
    background: "rgba(10,10,15,0.7)", border: "1px solid rgba(180,155,80,0.08)",
    borderRadius: 6, padding: small ? "8px 12px" : "12px 16px", minWidth: small ? 100 : 120, flex: "0 0 auto"
  }}>
    <div style={{ fontSize: 9, color: "#7a7a8a", letterSpacing: 0.8, textTransform: "uppercase", marginBottom: 2 }}>{label}</div>
    <div style={{ fontSize: small ? 14 : 19, fontWeight: 700, color, fontFamily: "'JetBrains Mono', monospace" }}>{value}</div>
    {sub && <div style={{ fontSize: 9, color: "#555", marginTop: 1 }}>{sub}</div>}
  </div>
);

const TabBtn = ({ active, label, onClick }) => (
  <button onClick={onClick} style={{
    background: active ? "rgba(180,155,80,0.15)" : "transparent",
    color: active ? "#D4B96A" : "#666", border: active ? "1px solid rgba(180,155,80,0.3)" : "1px solid rgba(255,255,255,0.05)",
    borderRadius: 5, padding: "5px 14px", fontSize: 11, fontWeight: 500, cursor: "pointer", transition: "all 0.2s"
  }}>{label}</button>
);

const InputField = ({ label, value, onChange, step = 1, min, max, width = 90, suffix = "" }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
    <label style={{ fontSize: 9, color: "#777", letterSpacing: 0.5 }}>{label}</label>
    <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
      <input type="number" value={value} step={step} min={min} max={max}
        onChange={e => onChange(parseFloat(e.target.value) || 0)}
        style={{
          width, background: "#0c0c12", color: "#e0e0e8", border: "1px solid rgba(180,155,80,0.2)",
          borderRadius: 4, padding: "5px 8px", fontSize: 12, fontFamily: "'JetBrains Mono', monospace"
        }}
      />
      {suffix && <span style={{ fontSize: 10, color: "#666" }}>{suffix}</span>}
    </div>
  </div>
);

// ═══════════════════════════════════════════════════════════════════════
// SVG CHARTS
// ═══════════════════════════════════════════════════════════════════════

const HeatmapSVG = ({ data, xLabels, yLabels, width = 640, height = 280, colorScheme = "diverging" }) => {
  const rows = data.length, cols = data[0].length;
  const cellW = (width - 60) / cols, cellH = (height - 40) / rows;
  const allVals = data.flat();
  const minV = Math.min(...allVals), maxV = Math.max(...allVals);
  const getColor = (v) => {
    if (colorScheme === "diverging") {
      if (v >= 0) { const t = Math.min(1, v / (maxV || 1)); return `rgb(${30 + 20 * t}, ${60 + 140 * t}, ${40 + 20 * t})`; }
      else { const t = Math.min(1, Math.abs(v) / (Math.abs(minV) || 1)); return `rgb(${60 + 160 * t}, ${30 + 20 * t}, ${30 + 10 * t})`; }
    } else { const t = (v - minV) / (maxV - minV + 0.001); return `rgb(${20 + 40 * t}, ${40 + 160 * t}, ${80 + 100 * t})`; }
  };
  return (
    <svg width={width} height={height}>
      {data.map((row, ri) => row.map((val, ci) => (
        <rect key={`${ri}-${ci}`} x={60 + ci * cellW} y={ri * cellH} width={cellW - 1} height={cellH - 1}
          fill={getColor(val)} rx={2}><title>{`${yLabels[ri]}% × ${xLabels[ci]}: ${val.toFixed(4)}`}</title></rect>
      )))}
      {xLabels.filter((_, i) => i % Math.ceil(cols / 8) === 0).map((l, idx) => {
        const i = idx * Math.ceil(cols / 8);
        return <text key={`x-${i}`} x={60 + i * cellW + cellW / 2} y={height - 5} fill="#666" fontSize={8} textAnchor="middle">{l}</text>;
      })}
      {yLabels.map((l, i) => (
        <text key={`y-${i}`} x={55} y={i * cellH + cellH / 2 + 3} fill="#666" fontSize={8} textAnchor="end">{l}%</text>
      ))}
    </svg>
  );
};

const LinePlotSVG = ({ datasets, width = 640, height = 200 }) => {
  const allY = datasets.flatMap(d => d.data.map(p => p.y));
  const allX = datasets[0].data.map(p => p.x);
  const minY = Math.min(...allY), maxY = Math.max(...allY);
  const minX = Math.min(...allX), maxX = Math.max(...allX);
  const padL = 55, padR = 10, padT = 10, padB = 30;
  const w = width - padL - padR, h = height - padT - padB;
  const sx = (x) => padL + ((x - minX) / (maxX - minX || 1)) * w;
  const sy = (y) => padT + h - ((y - minY) / (maxY - minY || 1)) * h;
  return (
    <svg width={width} height={height}>
      {[0, 0.25, 0.5, 0.75, 1].map(t => {
        const y = padT + h * (1 - t); const val = minY + t * (maxY - minY);
        return <g key={t}><line x1={padL} y1={y} x2={width - padR} y2={y} stroke="rgba(255,255,255,0.04)" />
          <text x={padL - 5} y={y + 3} fill="#555" fontSize={8} textAnchor="end">{val.toFixed(val < 10 ? 2 : 0)}</text></g>;
      })}
      {minY < 0 && maxY > 0 && <line x1={padL} y1={sy(0)} x2={width - padR} y2={sy(0)} stroke="rgba(255,255,255,0.12)" strokeDasharray="4,3" />}
      {datasets.map((ds, di) => {
        const points = ds.data.map(p => `${sx(p.x)},${sy(p.y)}`).join(" ");
        return <polyline key={di} points={points} fill="none" stroke={ds.color} strokeWidth={1.8} opacity={0.85} />;
      })}
      {datasets.map((ds, di) => (
        <g key={`leg-${di}`}>
          <line x1={padL + di * 140} y1={height - 8} x2={padL + di * 140 + 20} y2={height - 8} stroke={ds.color} strokeWidth={2} />
          <text x={padL + di * 140 + 25} y={height - 4} fill="#888" fontSize={9}>{ds.label}</text>
        </g>
      ))}
    </svg>
  );
};

const SparkLine = ({ data, dataKey, width = 200, height = 50, color = "#C9A84C" }) => {
  const vals = data.map(d => d[dataKey]);
  const min = Math.min(...vals), max = Math.max(...vals);
  const points = data.map((d, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - 5 - ((d[dataKey] - min) / (max - min + 0.0001)) * (height - 10);
    return `${x},${y}`;
  }).join(" ");
  return <svg width={width} height={height}><polyline points={points} fill="none" stroke={color} strokeWidth={1.5} /></svg>;
};

// ═══════════════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════════════

export default function UniversalQuantDashboard() {
  // ─── Market Parameters ───
  const [underlying, setUnderlying] = useState("Or (Gold)");
  const [symbol, setSymbol] = useState("XAU");
  const [optType, setOptType] = useState("call");
  const [spot, setSpot] = useState(5050);
  const [strike, setStrike] = useState(5150);
  const [vol, setVol] = useState(21.34);
  const [rate, setRate] = useState(4.5);
  const [maturity, setMaturity] = useState(5);
  const [numSims, setNumSims] = useState(10000);

  // ─── Heston Parameters ───
  const [hParams, setHP] = useState({ kappa: 2.0, theta: 0.045, xi: 0.5, rho: -0.7, v0: 0.0456 });

  // ─── State ───
  const [activeTab, setActiveTab] = useState("pricing");
  const [surfaceMetric, setSurfaceMetric] = useState("delta");
  const [selectedStructure, setSelectedStructure] = useState("vanilla");
  const [simKey, setSimKey] = useState(0);
  const [configOpen, setConfigOpen] = useState(true);
  // ─── Decision Aid State ───
  const [portfolio, setPortfolio] = useState(100000);
  const [maxRiskPct, setMaxRiskPct] = useState(2);
  const [targetReturn, setTargetReturn] = useState(50);
  const [conviction, setConviction] = useState(7);
  const [horizonMatch, setHorizonMatch] = useState(8);
  const [volView, setVolView] = useState(5);
  const [whatIfSpot, setWhatIfSpot] = useState(null);
  const [whatIfVol, setWhatIfVol] = useState(null);
  const [whatIfDays, setWhatIfDays] = useState(null);
  const [openSections, setOpenSections] = useState({});

  // derived
  const S = spot, K = strike, T = maturity / 12, r = rate / 100, sigma = vol / 100;
  const isCall = optType === "call";

  const applyPreset = (preset) => {
    setUnderlying(preset.name); setSymbol(preset.symbol);
    setSpot(preset.spot); setStrike(preset.strike);
    setVol(preset.vol); setRate(preset.rate);
    setMaturity(preset.maturity); setOptType(preset.type);
    setHP(prev => ({ ...prev, v0: (preset.vol / 100) ** 2, theta: (preset.vol / 100) ** 2 * 1.1 }));
    setSimKey(k => k + 1);
  };

  // ──── COMPUTATIONS ────

  const bs = useMemo(() => blackScholes(S, K, T, r, sigma, optType), [S, K, T, r, sigma, optType]);
  const premium = bs.price;
  const breakeven = isCall ? K + premium : K - premium;
  const pctMove = isCall ? ((breakeven / S) - 1) * 100 : ((S - breakeven) / S) * 100;
  const intrinsic = isCall ? Math.max(S - K, 0) : Math.max(K - S, 0);
  const moneyness = isCall ? ((K / S - 1) * 100) : ((S / K - 1) * -100);
  const moneynessLabel = moneyness > 0.5 ? "OTM" : moneyness < -0.5 ? "ITM" : "ATM";

  const fmt = useCallback((v, dec = 2) => {
    if (S >= 10000) return v.toFixed(dec);
    if (S >= 100) return v.toFixed(dec);
    if (S >= 1) return v.toFixed(Math.max(dec, 3));
    return v.toFixed(Math.max(dec, 5));
  }, [S]);

  const fmtPrice = useCallback((v) => {
    if (S >= 1000) return `$${v.toFixed(2)}`;
    if (S >= 10) return `$${v.toFixed(2)}`;
    return `$${v.toFixed(4)}`;
  }, [S]);

  // Heston
  const heston = useMemo(() => {
    const { kappa, theta, xi, rho, v0 } = hParams;
    return hestonMC(S, K, T, r, v0, kappa, theta, xi, rho, Math.min(numSims, 12000), 100, optType);
  }, [S, K, T, r, hParams, numSims, optType, simKey]);

  // Risk
  const riskMetrics = useMemo(() => {
    const pnls = heston.terminals.map(st => {
      const payoff = isCall ? Math.max(st - K, 0) : Math.max(K - st, 0);
      return payoff * Math.exp(-r * T) - premium;
    });
    return { ...computeRiskMetrics(pnls, 0.95), pnls };
  }, [heston, premium]);

  const riskMetrics99 = useMemo(() => computeRiskMetrics(riskMetrics.pnls, 0.99), [riskMetrics]);

  // Greeks sensitivity
  const greeksSens = useMemo(() => {
    const spots = [];
    for (let s = S * 0.9; s <= S * 1.1; s += S * 0.005) spots.push(s);
    return spots.map(s => {
      const g = blackScholes(s, K, T, r, sigma, optType);
      return { spot: s, delta: g.delta, gamma: g.gamma * 100, theta: g.theta, vega: g.vega };
    });
  }, [S, K, T, r, sigma, optType]);

  // Time decay
  const totalDays = Math.round(T * 365);
  const timeDecay = useMemo(() => {
    const pts = [];
    for (let d = 0; d <= totalDays; d += Math.max(1, Math.floor(totalDays / 150))) {
      const tLeft = (totalDays - d) / 365;
      const p = blackScholes(S, K, tLeft > 0 ? tLeft : 0.0001, r, sigma, optType);
      pts.push({ day: d, value: p.price, theta: p.theta });
    }
    return pts;
  }, [S, K, T, r, sigma, optType, totalDays]);

  // Greeks surface
  const surface = useMemo(() => greeksSurface(S, K, T, r, surfaceMetric, optType), [S, K, T, r, surfaceMetric, optType]);

  // Structures
  const structures = useMemo(() => {
    const spots = [];
    const range = S * 0.25;
    for (let s = S - range; s <= S + range; s += range / 50) spots.push(Math.round(s * 100) / 100);
    const K2 = Math.round((K + S * 0.04) * 100) / 100;
    const Katm = S;
    const K_put = Math.round((S * 0.97) * 100) / 100;
    const K_ratio = Math.round((K + S * 0.03) * 100) / 100;
    const K_b1 = Math.round((S * 0.98) * 100) / 100;
    const K_b2 = K;
    const K_b3 = Math.round((K + (K - K_b1)) * 100) / 100;

    const premSell = blackScholes(S, K2, T, r, sigma, "call").price;
    const spreadCost = premium - premSell;
    const callATM = blackScholes(S, Katm, T, r, sigma, "call").price;
    const putATM = blackScholes(S, Katm, T, r, sigma, "put").price;
    const straddleCost = callATM + putATM;
    const putSold = blackScholes(S, K_put, T, r, sigma, "put").price;
    const rrCost = premium - putSold;
    const premRatio = blackScholes(S, K_ratio, T, r, sigma, "call").price;
    const ratioCost = premium - 2 * premRatio;
    const c1 = blackScholes(S, K_b1, T, r, sigma, "call").price;
    const c2 = blackScholes(S, K_b2, T, r, sigma, "call").price;
    const c3 = blackScholes(S, K_b3, T, r, sigma, "call").price;
    const buttCost = c1 - 2 * c2 + c3;

    const vanilla = spots.map(s => ({ spot: s, pnl: (isCall ? Math.max(s - K, 0) : Math.max(K - s, 0)) - premium }));
    const bullSpread = spots.map(s => ({ spot: s, pnl: Math.max(s - K, 0) - Math.max(s - K2, 0) - spreadCost }));
    const straddle = spots.map(s => ({ spot: s, pnl: Math.max(s - Katm, 0) + Math.max(Katm - s, 0) - straddleCost }));
    const riskRev = spots.map(s => ({ spot: s, pnl: Math.max(s - K, 0) - Math.max(K_put - s, 0) - rrCost }));
    const ratioSpread = spots.map(s => ({ spot: s, pnl: Math.max(s - K, 0) - 2 * Math.max(s - K_ratio, 0) - ratioCost }));
    const butterfly = spots.map(s => ({ spot: s, pnl: Math.max(s - K_b1, 0) - 2 * Math.max(s - K_b2, 0) + Math.max(s - K_b3, 0) - buttCost }));

    return { spots, vanilla, bullSpread, straddle, riskRev, ratioSpread, butterfly, K2, Katm, K_put, K_ratio, K_b1, K_b2, K_b3, spreadCost, straddleCost, rrCost, ratioCost, buttCost };
  }, [S, K, T, r, sigma, premium, optType]);

  // Scenarios
  const scenarios = useMemo(() => {
    const shocks = [
      { name: "Base Case", spotChg: 0, volChg: 0 },
      { name: "Rally +5%", spotChg: 0.05, volChg: -0.02 },
      { name: "Rally +10%", spotChg: 0.10, volChg: -0.03 },
      { name: "Crash -5%", spotChg: -0.05, volChg: 0.05 },
      { name: "Crash -10%", spotChg: -0.10, volChg: 0.08 },
      { name: "Vol Spike +10pp", spotChg: 0, volChg: 0.10 },
      { name: "Vol Crush -5pp", spotChg: 0, volChg: -0.05 },
      { name: "Time -1M", spotChg: 0, volChg: 0, td: 1 / 12 },
      { name: "Time -3M", spotChg: 0, volChg: 0, td: 3 / 12 },
      { name: "Rally+5% + VolCrush", spotChg: 0.05, volChg: -0.05 },
      { name: "Crash-5% + VolSpike", spotChg: -0.05, volChg: 0.08 },
    ];
    return shocks.map(sc => {
      const newS = S * (1 + sc.spotChg);
      const newVol = Math.max(0.01, sigma + sc.volChg);
      const newT = Math.max(0.001, T - (sc.td || 0));
      const np = blackScholes(newS, K, newT, r, newVol, optType);
      return { ...sc, newPrice: np.price, pnl: np.price - premium, pnlPct: ((np.price - premium) / premium * 100), delta: np.delta };
    });
  }, [S, K, T, r, sigma, premium, optType]);

  // P&L histogram
  const pnlHistogram = useMemo(() => {
    const pnls = riskMetrics.pnls;
    const bins = 70;
    const min = Math.min(...pnls), max = Math.max(...pnls);
    const bw = (max - min) / bins;
    const hist = Array(bins).fill(0);
    pnls.forEach(v => { const idx = Math.min(bins - 1, Math.floor((v - min) / bw)); hist[idx]++; });
    return { hist, min, max, bw, maxCount: Math.max(...hist) };
  }, [riskMetrics]);

  // Vol smile & term
  const volSmile = useMemo(() => {
    const strikes = [];
    for (let k = S * 0.85; k <= S * 1.15; k += S * 0.01) strikes.push(Math.round(k * 100) / 100);
    return strikes.map(k => {
      const m = Math.log(k / S);
      const skew = vol + (-5) * m + 25 * m * m + (Math.sin(k * 7) * 0.15);
      return { strike: k, vol: Math.max(8, skew) };
    });
  }, [S, vol]);

  const termStructure = useMemo(() => {
    const mats = [0.5, 1, 2, 3, 4, 5, 6, 9, 12, 18, 24];
    return mats.map(m => ({
      maturity: m, vol: vol - 3 * Math.exp(-0.3 * m) + 1.5 * Math.log(1 + m) + Math.sin(m * 2) * 0.3
    }));
  }, [vol]);

  // ═══════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════

  const accent = isCall ? "#B49B50" : "#C07050";

  return (
    <div style={{
      minHeight: "100vh",
      background: isCall
        ? "radial-gradient(ellipse at 20% 0%, rgba(30,25,15,1) 0%, #08080d 60%)"
        : "radial-gradient(ellipse at 20% 0%, rgba(35,18,15,1) 0%, #08080d 60%)",
      color: "#ddd", fontFamily: "'Inter', -apple-system, sans-serif", padding: "16px 20px",
    }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
        ::-webkit-scrollbar{width:5px} ::-webkit-scrollbar-thumb{background:rgba(180,155,80,0.2);border-radius:3px}
        input[type=number]::-webkit-inner-spin-button{opacity:1}`}</style>

      {/* ═══ HEADER ═══ */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
        <div>
          <div style={{ fontSize: 10, color: accent, letterSpacing: 3 }}>QUANT DASHBOARD — UNIVERSEL</div>
          <h1 style={{ fontSize: 22, fontWeight: 700, margin: "2px 0", color: "#f0f0f5", fontFamily: "'JetBrains Mono', monospace" }}>
            {optType.toUpperCase()} {symbol} — Strike {S >= 10 ? K : K.toFixed(4)}
          </h1>
          <div style={{ fontSize: 11, color: "#666" }}>
            {underlying} · S={S >= 10 ? S : S.toFixed(4)} · K={S >= 10 ? K : K.toFixed(4)} · σ={vol}% · T={maturity}M · r={rate}% · Prime={fmtPrice(premium)}
          </div>
        </div>
        <button onClick={() => setConfigOpen(!configOpen)} style={{
          background: configOpen ? `${accent}22` : "transparent", color: accent,
          border: `1px solid ${accent}44`, borderRadius: 6, padding: "7px 16px",
          fontSize: 11, fontWeight: 600, cursor: "pointer"
        }}>
          {configOpen ? "▲ Masquer Config" : "▼ Configuration"}
        </button>
      </div>

      {/* ═══ CONFIG PANEL ═══ */}
      {configOpen && (
        <div style={{
          background: "rgba(12,12,18,0.95)", border: `1px solid ${accent}18`,
          borderRadius: 10, padding: 16, marginBottom: 14
        }}>
          {/* Presets */}
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 10, color: "#777", marginBottom: 6, letterSpacing: 1 }}>PRESETS RAPIDES</div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {PRESETS.map(p => (
                <button key={p.name} onClick={() => applyPreset(p)} style={{
                  background: underlying === p.name ? `${accent}22` : "rgba(20,20,28,0.8)",
                  color: underlying === p.name ? accent : "#888",
                  border: underlying === p.name ? `1px solid ${accent}44` : "1px solid rgba(255,255,255,0.06)",
                  borderRadius: 5, padding: "5px 12px", fontSize: 10, cursor: "pointer", fontWeight: 500
                }}>{p.name}</button>
              ))}
            </div>
          </div>

          {/* Custom inputs */}
          <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "flex-end" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <label style={{ fontSize: 9, color: "#777" }}>Sous-jacent</label>
              <input type="text" value={underlying} onChange={e => setUnderlying(e.target.value)}
                style={{ width: 120, background: "#0c0c12", color: "#e0e0e8", border: `1px solid ${accent}33`, borderRadius: 4, padding: "5px 8px", fontSize: 12 }} />
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <label style={{ fontSize: 9, color: "#777" }}>Symbole</label>
              <input type="text" value={symbol} onChange={e => setSymbol(e.target.value)}
                style={{ width: 70, background: "#0c0c12", color: "#e0e0e8", border: `1px solid ${accent}33`, borderRadius: 4, padding: "5px 8px", fontSize: 12 }} />
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <label style={{ fontSize: 9, color: "#777" }}>Type</label>
              <div style={{ display: "flex", gap: 4 }}>
                {["call", "put"].map(t => (
                  <button key={t} onClick={() => setOptType(t)} style={{
                    background: optType === t ? (t === "call" ? "rgba(76,175,80,0.2)" : "rgba(244,67,54,0.2)") : "rgba(20,20,28,0.8)",
                    color: optType === t ? (t === "call" ? "#81C784" : "#E57373") : "#666",
                    border: optType === t ? `1px solid ${t === "call" ? "#81C784" : "#E57373"}44` : "1px solid rgba(255,255,255,0.06)",
                    borderRadius: 4, padding: "5px 14px", fontSize: 11, cursor: "pointer", fontWeight: 600, textTransform: "uppercase"
                  }}>{t}</button>
                ))}
              </div>
            </div>
            <InputField label="Spot" value={spot} onChange={setSpot} step={S >= 100 ? 10 : S >= 1 ? 0.5 : 0.001} />
            <InputField label="Strike" value={strike} onChange={setStrike} step={S >= 100 ? 10 : S >= 1 ? 0.5 : 0.001} />
            <InputField label="Vol Impl." value={vol} onChange={setVol} step={0.5} suffix="%" width={65} />
            <InputField label="Taux" value={rate} onChange={setRate} step={0.25} suffix="%" width={55} />
            <InputField label="Maturité" value={maturity} onChange={setMaturity} step={1} suffix="mois" width={55} />
            <InputField label="Simulations" value={numSims} onChange={setNumSims} step={1000} width={75} />
            <button onClick={() => setSimKey(k => k + 1)} style={{
              background: accent, color: "#0a0a0f", border: "none", borderRadius: 5,
              padding: "8px 18px", fontSize: 11, fontWeight: 700, cursor: "pointer", marginBottom: 1
            }}>▶ CALCULER</button>
          </div>

          {/* Heston params */}
          <div style={{ marginTop: 12, paddingTop: 10, borderTop: "1px solid rgba(255,255,255,0.04)" }}>
            <div style={{ fontSize: 10, color: "#777", marginBottom: 6, letterSpacing: 1 }}>PARAMÈTRES HESTON</div>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              {[
                { key: "kappa", label: "κ (mean rev.)", step: 0.1 },
                { key: "theta", label: "θ (var LT)", step: 0.005 },
                { key: "xi", label: "ξ (vol of vol)", step: 0.05 },
                { key: "rho", label: "ρ (corrél.)", step: 0.05 },
                { key: "v0", label: "v₀ (var init.)", step: 0.005 },
              ].map(p => (
                <InputField key={p.key} label={p.label} value={hParams[p.key]} step={p.step} width={65}
                  onChange={v => setHP(prev => ({ ...prev, [p.key]: v }))} />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ═══ TABS ═══ */}
      <div style={{ display: "flex", gap: 5, marginBottom: 14, flexWrap: "wrap" }}>
        {[
          ["pricing", "Pricing & Greeks"],
          ["heston", "Heston SV"],
          ["risk", "VaR / CVaR"],
          ["surface", "Nappes Greeks"],
          ["structures", "Structuration"],
          ["smile", "Vol Surface"],
          ["scenarios", "Stress Tests"],
          ["pnl", "Distribution P&L"],
          ["guide", "Guide & Légende"],
          ["decision", "Aide à la Décision"],
        ].map(([k, l]) => <TabBtn key={k} active={activeTab === k} label={l} onClick={() => setActiveTab(k)} />)}
      </div>

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* PRICING & GREEKS */}
      {/* ═══════════════════════════════════════════════════════════ */}
      {activeTab === "pricing" && (
        <>
          <Panel title={`Black-Scholes — ${optType.toUpperCase()} ${symbol}`} number="1" accent={accent}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginBottom: 14 }}>
              <Metric label="Prime BS" value={fmtPrice(premium)} sub={`${(premium / S * 100).toFixed(2)}% du spot`} color={accent} />
              <Metric label="Delta Δ" value={bs.delta.toFixed(4)} color="#64B5F6" sub={isCall ? "Long delta" : "Short delta"} />
              <Metric label="Gamma Γ" value={bs.gamma.toFixed(6)} color="#81C784" sub="Convexité" />
              <Metric label="Theta Θ" value={`${bs.theta.toFixed(2)}/j`} color="#E57373" sub={`${(bs.theta * 30).toFixed(1)}/mois`} />
              <Metric label="Vega ν" value={bs.vega.toFixed(2)} color="#FFB74D" sub="$/1% vol" />
              <Metric label="Rho ρ" value={bs.rho.toFixed(2)} color="#BA68C8" sub="$/1% taux" />
              <Metric label="Breakeven" value={fmtPrice(breakeven)} color="#FF8A65" sub={`${pctMove >= 0 ? "+" : ""}${pctMove.toFixed(2)}%`} />
              <Metric label="Moneyness" value={`${moneyness.toFixed(1)}% ${moneynessLabel}`} color={moneynessLabel === "ITM" ? "#81C784" : moneynessLabel === "OTM" ? "#E57373" : "#FFB74D"} />
              <Metric label="Intrinsèque" value={fmtPrice(intrinsic)} color="#4DD0E1" />
              <Metric label="Val. Temps" value={fmtPrice(premium - intrinsic)} color={accent} />
              <Metric label="Levier" value={`${(S / premium).toFixed(1)}x`} color="#BA68C8" />
            </div>

            <div style={{ fontSize: 10, color: "#666", lineHeight: 1.7, marginBottom: 14 }}>
              d₁ = {bs.d1?.toFixed(4)} · d₂ = {bs.d2?.toFixed(4)} · N(d₁) = {normCDF(bs.d1).toFixed(4)} · N(d₂) = {normCDF(bs.d2).toFixed(4)}
            </div>

            {/* Greeks sparklines */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 12 }}>
              {[
                { key: "delta", label: "Delta vs Spot", color: "#64B5F6" },
                { key: "gamma", label: "Gamma vs Spot (×100)", color: "#81C784" },
                { key: "theta", label: "Theta vs Spot", color: "#E57373" },
                { key: "vega", label: "Vega vs Spot", color: "#FFB74D" },
              ].map(g => (
                <div key={g.key}>
                  <div style={{ fontSize: 10, color: "#777", marginBottom: 3 }}>{g.label}</div>
                  <SparkLine data={greeksSens} dataKey={g.key} width={155} height={60} color={g.color} />
                </div>
              ))}
            </div>
          </Panel>

          {/* Time Decay */}
          <Panel title="Décroissance Temporelle (Theta Decay)" number="θ" accent={accent}>
            <svg width={660} height={160}>
              {timeDecay.map((d, i) => {
                if (i === 0) return null;
                const maxV = timeDecay[0].value || 1;
                const x1 = ((i - 1) / timeDecay.length) * 660, x2 = (i / timeDecay.length) * 660;
                const y1 = 150 - (timeDecay[i - 1].value / maxV) * 135, y2 = 150 - (d.value / maxV) * 135;
                return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={accent} strokeWidth={2} />;
              })}
              <text x={5} y={15} fill="#777" fontSize={9}>{fmtPrice(timeDecay[0]?.value || 0)}</text>
              <text x={330} y={158} fill="#555" fontSize={9} textAnchor="middle">Jours → Expiration ({totalDays}j)</text>
            </svg>
          </Panel>
        </>
      )}

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* HESTON */}
      {/* ═══════════════════════════════════════════════════════════ */}
      {activeTab === "heston" && (
        <Panel title="Heston — Volatilité Stochastique" number="H" accent={accent}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginBottom: 14 }}>
            <Metric label="Prix Heston" value={fmtPrice(heston.price)} sub={`BS: ${fmtPrice(premium)}`} color={accent} />
            <Metric label="Écart vs BS" value={fmtPrice(heston.price - premium)}
              sub={`${((heston.price - premium) / premium * 100).toFixed(1)}%`}
              color={heston.price > premium ? "#81C784" : "#E57373"} />
            <Metric label="P(ITM)" value={`${(heston.probITM * 100).toFixed(1)}%`} color="#64B5F6" />
            <Metric label="P5 / P95" value={`${fmt(heston.pct(0.05), 0)} / ${fmt(heston.pct(0.95), 0)}`} color="#FFB74D" />
          </div>

          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 10, color: "#777", marginBottom: 4 }}>Trajectoires de volatilité stochastique</div>
            <svg width={660} height={120}>
              {heston.volPaths.map((vp, i) => {
                const minV = 0, maxV = 60;
                const pts = vp.map((v, j) => `${(j / (vp.length - 1)) * 660},${110 - ((Math.min(v, maxV) - minV) / (maxV - minV)) * 100}`).join(" ");
                return <polyline key={i} points={pts} fill="none" stroke="rgba(186,104,200,0.12)" strokeWidth={0.8} />;
              })}
              <line x1={0} y1={110 - ((vol - 0) / 60) * 100} x2={660} y2={110 - ((vol - 0) / 60) * 100}
                stroke={accent} strokeWidth={1} strokeDasharray="4,3" />
            </svg>
          </div>

          <div>
            <div style={{ fontSize: 10, color: "#777", marginBottom: 4 }}>Trajectoires des prix</div>
            <svg width={660} height={180}>
              {(() => {
                const allV = heston.paths.flat();
                const mn = Math.min(...allV), mx = Math.max(...allV);
                return <>
                  {heston.paths.map((p, i) => {
                    const terminal = p[p.length - 1];
                    const itm = isCall ? terminal > K : terminal < K;
                    const col = itm ? "rgba(76,175,80,0.12)" : "rgba(244,67,54,0.08)";
                    const pts = p.map((v, j) => `${(j / (p.length - 1)) * 660},${170 - ((v - mn) / (mx - mn)) * 155}`).join(" ");
                    return <polyline key={i} points={pts} fill="none" stroke={col} strokeWidth={0.7} />;
                  })}
                  <line x1={0} y1={170 - ((K - mn) / (mx - mn)) * 155} x2={660} y2={170 - ((K - mn) / (mx - mn)) * 155}
                    stroke={accent} strokeWidth={1.5} strokeDasharray="6,4" />
                  <text x={5} y={170 - ((K - mn) / (mx - mn)) * 155 - 4} fill={accent} fontSize={9}>K={S >= 10 ? K : K.toFixed(4)}</text>
                </>;
              })()}
            </svg>
          </div>
        </Panel>
      )}

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* RISK */}
      {/* ═══════════════════════════════════════════════════════════ */}
      {activeTab === "risk" && (
        <Panel title="Value-at-Risk & Expected Shortfall" number="R" accent={accent}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginBottom: 16 }}>
            <Metric label="VaR 95%" value={fmtPrice(riskMetrics.VaR)} color="#E57373" sub="Perte max 95%" />
            <Metric label="CVaR 95%" value={fmtPrice(riskMetrics.CVaR)} color="#EF5350" sub="Expected Shortfall" />
            <Metric label="VaR 99%" value={fmtPrice(riskMetrics99.VaR)} color="#D32F2F" />
            <Metric label="CVaR 99%" value={fmtPrice(riskMetrics99.CVaR)} color="#B71C1C" />
            <Metric label="Perte Max" value={fmtPrice(riskMetrics.maxLoss)} color="#FF8A65" />
            <Metric label="Gain Max" value={fmtPrice(riskMetrics.maxGain)} color="#81C784" />
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <Metric small label="E[P&L]" value={fmtPrice(riskMetrics.mean)} color={riskMetrics.mean > 0 ? "#81C784" : "#E57373"} />
              <Metric small label="Écart-type" value={fmtPrice(riskMetrics.std)} color="#64B5F6" />
              <Metric small label="Skewness" value={riskMetrics.skew.toFixed(3)} color="#FFB74D" />
              <Metric small label="Kurtosis exc." value={riskMetrics.kurt.toFixed(3)} color="#BA68C8" />
            </div>
            <div>
              <div style={{ fontSize: 10, color: "#777", marginBottom: 4 }}>Distribution P&L avec VaR</div>
              <svg width={320} height={130}>
                {pnlHistogram.hist.map((count, i) => {
                  const x = 10 + (i / pnlHistogram.hist.length) * 300;
                  const h = (count / pnlHistogram.maxCount) * 110;
                  const val = pnlHistogram.min + (i + 0.5) * pnlHistogram.bw;
                  const isVaR = val < -riskMetrics.VaR;
                  return <rect key={i} x={x} y={120 - h} width={300 / pnlHistogram.hist.length - 0.5} height={h}
                    fill={isVaR ? "rgba(244,67,54,0.5)" : val > 0 ? "rgba(76,175,80,0.4)" : "rgba(100,100,120,0.3)"} rx={1} />;
                })}
              </svg>
            </div>
          </div>
        </Panel>
      )}

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* GREEKS SURFACE */}
      {/* ═══════════════════════════════════════════════════════════ */}
      {activeTab === "surface" && (
        <Panel title="Nappes de Sensibilité" number="G" accent={accent}>
          <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
            {["delta", "gamma", "theta", "vega"].map(m => (
              <TabBtn key={m} active={surfaceMetric === m} label={m.charAt(0).toUpperCase() + m.slice(1)} onClick={() => setSurfaceMetric(m)} />
            ))}
          </div>
          <div style={{ fontSize: 10, color: "#777", marginBottom: 6 }}>
            {surfaceMetric.charAt(0).toUpperCase() + surfaceMetric.slice(1)} — Spot (X) × Volatilité (Y)
          </div>
          <HeatmapSVG data={surface.surface} xLabels={surface.spots.map(String)} yLabels={surface.vols.map(String)}
            width={660} height={280} colorScheme={surfaceMetric === "theta" ? "diverging" : "sequential"} />
        </Panel>
      )}

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* STRUCTURED PRODUCTS */}
      {/* ═══════════════════════════════════════════════════════════ */}
      {activeTab === "structures" && (
        <Panel title="Stratégies Structurées" number="S" accent={accent}>
          <div style={{ display: "flex", gap: 5, marginBottom: 12, flexWrap: "wrap" }}>
            {[
              ["vanilla", `${optType === "call" ? "Call" : "Put"} Vanille`],
              ["bullSpread", "Bull Call Spread"],
              ["straddle", "Straddle ATM"],
              ["riskRev", "Risk Reversal"],
              ["ratioSpread", "Ratio 1×2"],
              ["butterfly", "Butterfly"],
            ].map(([k, l]) => <TabBtn key={k} active={selectedStructure === k} label={l} onClick={() => setSelectedStructure(k)} />)}
          </div>

          <div style={{ background: "rgba(10,10,15,0.6)", borderRadius: 6, padding: 12, marginBottom: 12, fontSize: 11, color: "#aaa", lineHeight: 1.7 }}>
            {selectedStructure === "vanilla" && <><b style={{ color: accent }}>{optType === "call" ? "Call" : "Put"} Vanille K={S >= 10 ? K : K.toFixed(4)}</b> — Coût: {fmtPrice(premium)} · Breakeven: {fmtPrice(breakeven)}</>}
            {selectedStructure === "bullSpread" && <><b style={{ color: accent }}>Bull Spread {S >= 10 ? K : K.toFixed(4)}/{S >= 10 ? structures.K2 : structures.K2.toFixed(4)}</b> — Coût net: {fmtPrice(structures.spreadCost)} · Gain max: {fmtPrice(structures.K2 - K - structures.spreadCost)}</>}
            {selectedStructure === "straddle" && <><b style={{ color: accent }}>Straddle ATM {S >= 10 ? S : S.toFixed(4)}</b> — Coût: {fmtPrice(structures.straddleCost)} · Pari sur la volatilité réalisée</>}
            {selectedStructure === "riskRev" && <><b style={{ color: accent }}>Risk Reversal</b> — Call {S >= 10 ? K : K.toFixed(4)} + Put vendu {S >= 10 ? structures.K_put : structures.K_put.toFixed(4)} · Net: {fmtPrice(structures.rrCost)}</>}
            {selectedStructure === "ratioSpread" && <><b style={{ color: accent }}>Ratio 1×2</b> — Achat 1 Call {S >= 10 ? K : K.toFixed(4)} + Vente 2 Calls {S >= 10 ? structures.K_ratio : structures.K_ratio.toFixed(4)}</>}
            {selectedStructure === "butterfly" && <><b style={{ color: accent }}>Butterfly</b> — {S >= 10 ? structures.K_b1 : structures.K_b1.toFixed(4)}/{S >= 10 ? structures.K_b2 : structures.K_b2.toFixed(4)}/{S >= 10 ? structures.K_b3 : structures.K_b3.toFixed(4)} · Coût: {fmtPrice(structures.buttCost)}</>}
          </div>

          {(() => {
            const dataMap = { vanilla: structures.vanilla, bullSpread: structures.bullSpread, straddle: structures.straddle, riskRev: structures.riskRev, ratioSpread: structures.ratioSpread, butterfly: structures.butterfly };
            return <LinePlotSVG width={660} height={220} datasets={[
              { data: dataMap[selectedStructure].map(d => ({ x: d.spot, y: d.pnl })), color: accent, label: selectedStructure },
              { data: structures.vanilla.map(d => ({ x: d.spot, y: d.pnl })), color: "rgba(100,181,246,0.3)", label: "Vanille (ref)" },
            ]} />;
          })()}
        </Panel>
      )}

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* VOL SMILE */}
      {/* ═══════════════════════════════════════════════════════════ */}
      {activeTab === "smile" && (
        <Panel title="Surface de Volatilité" number="V" accent={accent}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <div>
              <div style={{ fontSize: 10, color: "#777", marginBottom: 6 }}>Vol Smile (σ vs Strike)</div>
              <LinePlotSVG width={320} height={200} datasets={[
                { data: volSmile.map(d => ({ x: d.strike, y: d.vol })), color: "#BA68C8", label: "Vol Impl." }
              ]} />
            </div>
            <div>
              <div style={{ fontSize: 10, color: "#777", marginBottom: 6 }}>Term Structure</div>
              <LinePlotSVG width={320} height={200} datasets={[
                { data: termStructure.map(d => ({ x: d.maturity, y: d.vol })), color: "#FFB74D", label: "Vol (mois)" }
              ]} />
            </div>
          </div>
          <div style={{ marginTop: 12, fontSize: 10, color: "#666", lineHeight: 1.7 }}>
            <b style={{ color: accent }}>Position Vega:</b> Long vega de {bs.vega.toFixed(2)} — chaque +1% de vol ≈ +{fmtPrice(bs.vega)} sur la prime.
            Le skew implique un coût plus élevé pour les puts OTM (crash premium).
          </div>
        </Panel>
      )}

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* STRESS TESTS */}
      {/* ═══════════════════════════════════════════════════════════ */}
      {activeTab === "scenarios" && (
        <Panel title="Stress Tests Multi-facteurs" number="X" accent={accent}>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${accent}22` }}>
                  {["Scénario", "ΔSpot", "ΔVol", "Nvelle Prime", "P&L", "P&L %", "Delta"].map(h => (
                    <th key={h} style={{ padding: "7px 8px", color: "#777", fontWeight: 500, textAlign: "left" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {scenarios.map((sc, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.03)", background: sc.name === "Base Case" ? `${accent}08` : "transparent" }}>
                    <td style={{ padding: "5px 8px", color: "#ccc" }}>{sc.name}</td>
                    <td style={{ padding: "5px 8px", color: "#aaa", fontFamily: "monospace" }}>
                      {sc.spotChg ? `${sc.spotChg > 0 ? "+" : ""}${(sc.spotChg * 100).toFixed(0)}%` : sc.td ? `T-${(sc.td * 12).toFixed(0)}M` : "—"}
                    </td>
                    <td style={{ padding: "5px 8px", color: "#aaa", fontFamily: "monospace" }}>
                      {sc.volChg ? `${sc.volChg > 0 ? "+" : ""}${(sc.volChg * 100).toFixed(0)}pp` : "—"}
                    </td>
                    <td style={{ padding: "5px 8px", color: accent, fontFamily: "monospace" }}>{fmtPrice(sc.newPrice)}</td>
                    <td style={{ padding: "5px 8px", color: sc.pnl >= 0 ? "#81C784" : "#E57373", fontFamily: "monospace", fontWeight: 600 }}>
                      {sc.pnl >= 0 ? "+" : ""}{fmtPrice(sc.pnl).replace("$", "")}
                    </td>
                    <td style={{ padding: "5px 8px", color: sc.pnlPct >= 0 ? "#81C784" : "#E57373", fontFamily: "monospace" }}>
                      {sc.pnlPct >= 0 ? "+" : ""}{sc.pnlPct.toFixed(1)}%
                    </td>
                    <td style={{ padding: "5px 8px", color: "#64B5F6", fontFamily: "monospace" }}>{sc.delta.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ marginTop: 14, fontSize: 10, color: "#777" }}>Impact visuel</div>
          <svg width={660} height={180}>
            {scenarios.map((sc, i) => {
              const maxAbs = Math.max(...scenarios.map(s => Math.abs(s.pnl)));
              const barH = maxAbs > 0 ? (Math.abs(sc.pnl) / maxAbs) * 70 : 0;
              const isPos = sc.pnl >= 0;
              const x = 25 + i * 56;
              return (
                <g key={i}>
                  <rect x={x} y={isPos ? 85 - barH : 85} width={42} height={barH}
                    fill={isPos ? "rgba(76,175,80,0.5)" : "rgba(244,67,54,0.4)"} rx={3} />
                  <text x={x + 21} y={isPos ? 80 - barH : 85 + barH + 12}
                    fill={isPos ? "#81C784" : "#E57373"} fontSize={7} textAnchor="middle" fontFamily="monospace">
                    {sc.pnlPct.toFixed(0)}%
                  </text>
                  <text x={x + 21} y={170} fill="#555" fontSize={6} textAnchor="middle"
                    transform={`rotate(-40, ${x + 21}, 170)`}>{sc.name.substring(0, 14)}</text>
                </g>
              );
            })}
            <line x1={25} y1={85} x2={640} y2={85} stroke="rgba(255,255,255,0.08)" />
          </svg>
        </Panel>
      )}

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* P&L DISTRIBUTION */}
      {/* ═══════════════════════════════════════════════════════════ */}
      {activeTab === "pnl" && (
        <Panel title="Distribution Complète du P&L" number="D" accent={accent}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginBottom: 14 }}>
            <Metric label="E[P&L]" value={fmtPrice(riskMetrics.mean)} color={riskMetrics.mean > 0 ? "#81C784" : "#E57373"} />
            <Metric label="Médiane" value={fmtPrice(riskMetrics.sorted[Math.floor(riskMetrics.sorted.length / 2)])} color="#64B5F6" />
            <Metric label="Skew" value={riskMetrics.skew.toFixed(2)} color="#FFB74D" sub={riskMetrics.skew > 0 ? "Queue droite ↗" : "Queue gauche"} />
            <Metric label="% Profit" value={`${(riskMetrics.pnls.filter(p => p > 0).length / riskMetrics.pnls.length * 100).toFixed(1)}%`} color="#81C784" />
            <Metric label="% > 2× prime" value={`${(riskMetrics.pnls.filter(p => p > premium).length / riskMetrics.pnls.length * 100).toFixed(1)}%`} color="#4DD0E1" />
          </div>

          <svg width={660} height={180}>
            {pnlHistogram.hist.map((count, i) => {
              const x = 20 + (i / pnlHistogram.hist.length) * 620;
              const h = (count / pnlHistogram.maxCount) * 160;
              const val = pnlHistogram.min + (i + 0.5) * pnlHistogram.bw;
              return <rect key={i} x={x} y={170 - h} width={620 / pnlHistogram.hist.length - 0.5} height={h}
                fill={val > 0 ? "rgba(76,175,80,0.5)" : "rgba(244,67,54,0.35)"} rx={1} />;
            })}
            {(() => {
              const zx = 20 + ((0 - pnlHistogram.min) / (pnlHistogram.max - pnlHistogram.min)) * 620;
              return <><line x1={zx} y1={0} x2={zx} y2={170} stroke="rgba(255,255,255,0.3)" strokeWidth={1.5} />
                <text x={zx + 3} y={12} fill="#aaa" fontSize={8}>BE</text></>;
            })()}
          </svg>
        </Panel>
      )}

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* GUIDE & LÉGENDE */}
      {/* ═══════════════════════════════════════════════════════════ */}
      {activeTab === "guide" && (() => {
        const Entry = ({ icon, title, color, children, formula }) => (
          <div style={{ background: "rgba(10,10,15,0.6)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: 8, padding: "16px 18px", marginBottom: 10 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
              <div style={{ width: 36, height: 36, borderRadius: "50%", background: `${color}18`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, fontWeight: 700, color, flexShrink: 0 }}>{icon}</div>
              <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700, color }}>{title}</h3>
            </div>
            <div style={{ fontSize: 12, color: "#bbb", lineHeight: 1.85, paddingLeft: 46 }}>
              {children}
              {formula && (
                <div style={{ marginTop: 8, background: "rgba(0,0,0,0.3)", borderRadius: 5, padding: "8px 12px", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#999", overflowX: "auto" }}>
                  {formula}
                </div>
              )}
            </div>
          </div>
        );

        const SubEntry = ({ label, color = "#aaa", children }) => (
          <div style={{ marginTop: 8, paddingLeft: 12, borderLeft: `2px solid ${color}33` }}>
            <span style={{ color, fontWeight: 600, fontSize: 12 }}>{label}</span>
            <div style={{ fontSize: 11, color: "#999", marginTop: 2, lineHeight: 1.7 }}>{children}</div>
          </div>
        );

        const YourVal = ({ label, value, color = accent }) => (
          <span style={{ display: "inline-flex", alignItems: "center", gap: 4, background: `${color}12`, border: `1px solid ${color}33`, borderRadius: 4, padding: "2px 8px", fontSize: 10, color, fontFamily: "'JetBrains Mono', monospace", margin: "2px 4px 2px 0" }}>
            {label}: <b>{value}</b>
          </span>
        );

        return (
          <>
            <Panel title="Encyclopédie — Tout comprendre" number="?" accent={accent}>
              <div style={{ fontSize: 11, color: "#777", marginBottom: 14, lineHeight: 1.6 }}>
                Chaque concept est expliqué avec sa définition, sa formule, une analogie simple, et <b style={{ color: accent }}>sa valeur actuelle pour votre position</b>.
              </div>

              {/* ── BLACK-SCHOLES ── */}
              <Entry icon="BS" title="Modèle de Black-Scholes" color="#D4B96A"
                formula="C = S·N(d₁) − K·e^(−rT)·N(d₂)   où   d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)">
                <p style={{ margin: "0 0 6px" }}>Le modèle de référence pour évaluer le prix théorique d'une option européenne. Il part de 5 inputs (spot, strike, vol, taux, maturité) et donne un prix « juste » en supposant que les rendements suivent une loi log-normale.</p>
                <p style={{ margin: "0 0 6px" }}><b style={{ color: "#e0e0e0" }}>Analogie :</b> C'est comme une calculatrice d'assurance — elle estime combien vaut la « protection » (ou le « pari ») que représente l'option, en pesant la probabilité que le prix finisse favorablement.</p>
                <p style={{ margin: "0 0 6px" }}><b style={{ color: "#e0e0e0" }}>Limites :</b> Suppose une volatilité constante (pas de smile), pas de sauts de prix, des marchés continus. C'est pourquoi on complète avec Heston.</p>
                <div style={{ marginTop: 8 }}>
                  <YourVal label="Prime BS" value={fmtPrice(premium)} />
                  <YourVal label="d₁" value={bs.d1?.toFixed(4)} />
                  <YourVal label="d₂" value={bs.d2?.toFixed(4)} />
                  <YourVal label="N(d₁)" value={normCDF(bs.d1).toFixed(4)} />
                </div>
              </Entry>

              {/* ── LES GRECS ── */}
              <Entry icon="Δ" title="Les Greeks (Lettres Grecques)" color="#64B5F6">
                <p style={{ margin: "0 0 8px" }}>Les Greeks mesurent la sensibilité du prix de l'option à différents facteurs. Ce sont les « capteurs » de votre position — ils vous disent comment elle réagira à chaque type de mouvement du marché.</p>

                <SubEntry label="Delta (Δ) — Sensibilité au spot" color="#64B5F6">
                  <p style={{ margin: "2px 0" }}>De combien bouge le prix de l'option quand le sous-jacent bouge de 1$. Un delta de 0.40 signifie que si l'or monte de $100, votre option prend ~$40.</p>
                  <p style={{ margin: "2px 0" }}><b>Analogie :</b> C'est le « compteur de vitesse » de votre option. Delta = 0.5 → vous êtes exposé comme si vous déteniez la moitié du sous-jacent.</p>
                  <p style={{ margin: "2px 0" }}><b>Aussi interprété comme :</b> probabilité approximative que l'option expire ITM.</p>
                  <YourVal label="Votre Δ" value={bs.delta.toFixed(4)} color="#64B5F6" />
                  <YourVal label="Équivalent" value={`${(bs.delta * 100).toFixed(1)}% du sous-jacent`} color="#64B5F6" />
                </SubEntry>

                <SubEntry label="Gamma (Γ) — Accélération du Delta" color="#81C784">
                  <p style={{ margin: "2px 0" }}>De combien le Delta change quand le spot bouge de 1$. Un gamma élevé = votre delta change vite, ce qui est à la fois une opportunité et un risque.</p>
                  <p style={{ margin: "2px 0" }}><b>Analogie :</b> Si Delta est la vitesse, Gamma est l'accélération. Fort Gamma = votre exposition change rapidement.</p>
                  <p style={{ margin: "2px 0" }}><b>Quand c'est important :</b> Maximal quand l'option est ATM et proche de l'expiration. Les market makers surveillent le gamma en permanence.</p>
                  <YourVal label="Votre Γ" value={bs.gamma.toFixed(6)} color="#81C784" />
                </SubEntry>

                <SubEntry label="Theta (Θ) — Érosion temporelle" color="#E57373">
                  <p style={{ margin: "2px 0" }}>Combien votre option perd de valeur chaque jour qui passe, toutes choses égales par ailleurs. C'est le « loyer » que vous payez pour détenir l'option.</p>
                  <p style={{ margin: "2px 0" }}><b>Analogie :</b> Comme un glaçon qui fond — chaque jour, un peu de valeur temps s'évapore. Et ça accélère en fin de vie (les 30 derniers jours sont les plus destructeurs).</p>
                  <p style={{ margin: "2px 0" }}><b>Règle clé :</b> Acheteur d'option = Theta négatif (ça joue contre vous). Vendeur = Theta positif (ça joue pour vous).</p>
                  <YourVal label="Votre Θ" value={`${bs.theta.toFixed(2)}/jour`} color="#E57373" />
                  <YourVal label="Perte/mois" value={fmtPrice(Math.abs(bs.theta * 30))} color="#E57373" />
                </SubEntry>

                <SubEntry label="Vega (ν) — Sensibilité à la volatilité" color="#FFB74D">
                  <p style={{ margin: "2px 0" }}>De combien le prix de l'option change si la volatilité implicite bouge de 1 point. Vous êtes « long vega » = vous profitez si la vol monte.</p>
                  <p style={{ margin: "2px 0" }}><b>Analogie :</b> C'est votre « pari sur l'incertitude ». Plus le marché a peur (vol haute), plus votre option vaut cher. Si le calme revient, votre option perd de la valeur même si le spot ne bouge pas.</p>
                  <YourVal label="Votre ν" value={`${bs.vega.toFixed(2)}/1% vol`} color="#FFB74D" />
                  <YourVal label="Si vol +5%" value={`+${fmtPrice(bs.vega * 5)}`} color="#FFB74D" />
                </SubEntry>

                <SubEntry label="Rho (ρ) — Sensibilité aux taux" color="#BA68C8">
                  <p style={{ margin: "2px 0" }}>Impact d'un changement de 1% du taux sans risque. Généralement le grec le moins impactant, sauf sur les longues maturités ou les marchés de taux.</p>
                  <YourVal label="Votre ρ" value={bs.rho.toFixed(2)} color="#BA68C8" />
                </SubEntry>
              </Entry>

              {/* ── HESTON ── */}
              <Entry icon="H" title="Modèle de Heston (Volatilité Stochastique)" color="#BA68C8"
                formula="dS = μS·dt + √v·S·dW₁   |   dv = κ(θ−v)dt + ξ√v·dW₂   |   corr(dW₁,dW₂) = ρ">
                <p style={{ margin: "0 0 6px" }}>Contrairement à Black-Scholes qui suppose une volatilité fixe, Heston laisse la volatilité elle-même fluctuer aléatoirement. C'est plus réaliste car dans la vraie vie, la vol n'est jamais constante.</p>
                <p style={{ margin: "0 0 6px" }}><b style={{ color: "#e0e0e0" }}>Les 5 paramètres :</b></p>

                <SubEntry label="κ (kappa) — Vitesse de retour à la moyenne" color="#BA68C8">
                  Si la vol s'écarte de sa moyenne long-terme, κ contrôle à quelle vitesse elle y revient. κ élevé = la vol revient vite à la normale.
                  <br /><YourVal label="κ" value={hParams.kappa} color="#BA68C8" />
                </SubEntry>
                <SubEntry label="θ (theta Heston) — Variance long-terme" color="#BA68C8">
                  Le niveau « normal » de variance vers lequel le processus tend. √θ ≈ volatilité long-terme.
                  <br /><YourVal label="θ" value={hParams.theta} color="#BA68C8" />
                  <YourVal label="Vol LT" value={`${(Math.sqrt(hParams.theta) * 100).toFixed(1)}%`} color="#BA68C8" />
                </SubEntry>
                <SubEntry label="ξ (xi) — Vol-of-Vol" color="#BA68C8">
                  À quel point la volatilité elle-même est volatile. ξ élevé = queues de distribution plus épaisses, smile de vol plus prononcé.
                  <br /><YourVal label="ξ" value={hParams.xi} color="#BA68C8" />
                </SubEntry>
                <SubEntry label="ρ (rho Heston) — Corrélation Spot-Vol" color="#BA68C8">
                  Lien entre les mouvements du prix et de la vol. Négatif = quand le prix baisse, la vol monte (effet de levier classique sur les actions/commodités).
                  <br /><YourVal label="ρ" value={hParams.rho} color="#BA68C8" />
                </SubEntry>
                <SubEntry label="v₀ — Variance initiale" color="#BA68C8">
                  Le point de départ de la variance. √v₀ = volatilité initiale.
                  <br /><YourVal label="v₀" value={hParams.v0} color="#BA68C8" />
                  <YourVal label="Vol init." value={`${(Math.sqrt(hParams.v0) * 100).toFixed(1)}%`} color="#BA68C8" />
                </SubEntry>
              </Entry>

              {/* ── MONTE CARLO ── */}
              <Entry icon="MC" title="Simulation Monte Carlo" color="#4DD0E1">
                <p style={{ margin: "0 0 6px" }}>Technique de simulation numérique : on génère des milliers de scénarios aléatoires d'évolution du prix, on calcule le payoff dans chaque scénario, puis on fait la moyenne actualisée pour obtenir le prix.</p>
                <p style={{ margin: "0 0 6px" }}><b style={{ color: "#e0e0e0" }}>Analogie :</b> Imaginez que vous jouiez la même partie 10 000 fois. Certaines fois l'or monte, d'autres il baisse. En moyennant vos gains sur toutes les parties, vous obtenez la « valeur espérée » de votre position.</p>
                <p style={{ margin: "0 0 6px" }}><b style={{ color: "#e0e0e0" }}>Pourquoi c'est utile :</b> Permet de pricer des options exotiques, de capturer la distribution complète des résultats (pas juste un prix moyen), et de tester des modèles complexes comme Heston.</p>
                <p style={{ margin: 0 }}><b style={{ color: "#e0e0e0" }}>Précision :</b> L'erreur standard diminue en 1/√N — pour diviser l'erreur par 2, il faut 4× plus de simulations.</p>
                <div style={{ marginTop: 6 }}>
                  <YourVal label="Simulations" value={numSims.toLocaleString()} color="#4DD0E1" />
                  <YourVal label="Prix MC (Heston)" value={fmtPrice(heston.price)} color="#4DD0E1" />
                  <YourVal label="P(ITM)" value={`${(heston.probITM * 100).toFixed(1)}%`} color="#4DD0E1" />
                </div>
              </Entry>

              {/* ── VaR / CVaR ── */}
              <Entry icon="V" title="VaR & CVaR (Expected Shortfall)" color="#EF5350">
                <p style={{ margin: "0 0 6px" }}><b style={{ color: "#E57373" }}>Value-at-Risk (VaR) :</b> La perte maximale que vous pouvez subir avec un certain niveau de confiance. « VaR 95% = $X » signifie que dans 95% des scénarios, votre perte ne dépasse pas $X.</p>
                <p style={{ margin: "0 0 6px" }}><b style={{ color: "#E57373" }}>CVaR (Conditional VaR / Expected Shortfall) :</b> Plus conservateur — c'est la perte moyenne dans les 5% de pires scénarios (la queue de distribution). Répond à « si ça tourne mal, à quel point ça tourne mal ? »</p>
                <p style={{ margin: "0 0 6px" }}><b style={{ color: "#e0e0e0" }}>Analogie :</b> La VaR est le seuil d'inondation pour une crue centennale. La CVaR est la hauteur moyenne de l'eau quand cette crue se produit effectivement.</p>
                <p style={{ margin: "0 0 6px" }}><b style={{ color: "#e0e0e0" }}>Pour une option achetée :</b> La VaR est souvent proche de la prime (car la perte max = prime), mais la CVaR peut révéler à quel point la perte totale est fréquente.</p>
                <div style={{ marginTop: 6 }}>
                  <YourVal label="VaR 95%" value={fmtPrice(riskMetrics.VaR)} color="#EF5350" />
                  <YourVal label="CVaR 95%" value={fmtPrice(riskMetrics.CVaR)} color="#EF5350" />
                  <YourVal label="Perte max" value={fmtPrice(riskMetrics.maxLoss)} color="#EF5350" />
                </div>
              </Entry>

              {/* ── SKEW & KURTOSIS ── */}
              <Entry icon="SK" title="Skewness & Kurtosis" color="#FFB74D">
                <SubEntry label="Skewness (Asymétrie)" color="#FFB74D">
                  <p style={{ margin: "2px 0" }}>Mesure l'asymétrie de la distribution. Skew positif = queue droite allongée (potentiel de gains extrêmes). Skew négatif = queue gauche (risque de pertes extrêmes). Un call long a typiquement un skew positif : beaucoup de petites pertes (prime perdue) et quelques gros gains.</p>
                  <YourVal label="Skew" value={riskMetrics.skew.toFixed(3)} color="#FFB74D" />
                </SubEntry>
                <SubEntry label="Kurtosis (Épaisseur des queues)" color="#FFB74D">
                  <p style={{ margin: "2px 0" }}>Mesure l'épaisseur des queues par rapport à une gaussienne. Kurtosis excédentaire {">"} 0 = événements extrêmes plus fréquents que prévu (queues « grasses »). C'est crucial car Black-Scholes sous-estime ces événements.</p>
                  <YourVal label="Kurt. exc." value={riskMetrics.kurt.toFixed(3)} color="#FFB74D" />
                </SubEntry>
              </Entry>

              {/* ── VOL SMILE ── */}
              <Entry icon="😊" title="Smile de Volatilité & Surface" color="#CE93D8">
                <p style={{ margin: "0 0 6px" }}>En théorie (Black-Scholes), la vol implicite devrait être la même pour tous les strikes. En pratique, elle forme un « sourire » : plus élevée pour les strikes éloignés (surtout les puts OTM), car le marché price un risque de crash.</p>
                <p style={{ margin: "0 0 6px" }}><b style={{ color: "#e0e0e0" }}>Skew (pente du smile) :</b> Le côté gauche du smile est généralement plus haut — les puts OTM coûtent relativement plus cher car le marché a peur des crashs.</p>
                <p style={{ margin: 0 }}><b style={{ color: "#e0e0e0" }}>Term structure :</b> La vol varie aussi selon la maturité. En contango (normal) : vol long-terme {">"} vol court-terme. En backwardation (stress) : l'inverse.</p>
              </Entry>

              {/* ── STRUCTURES ── */}
              <Entry icon="📐" title="Produits Structurés — Les Stratégies" color="#81C784">
                <SubEntry label="Bull Call Spread" color="#81C784">Achat d'un Call + Vente d'un Call à strike plus élevé. Réduit le coût en sacrifiant le potentiel illimité. Idéal pour une vue haussière modérée.</SubEntry>
                <SubEntry label="Straddle" color="#81C784">Achat d'un Call + Put au même strike. Pari pur sur la volatilité, neutre en direction. Profite si le marché fait un grand mouvement (dans n'importe quel sens).</SubEntry>
                <SubEntry label="Risk Reversal" color="#81C784">Vente d'un Put OTM + Achat d'un Call OTM. Le put vendu finance le call. Structure « zéro coût » mais avec risque de perte si le sous-jacent baisse fortement.</SubEntry>
                <SubEntry label="Ratio Spread 1×2" color="#81C784">Achat 1 Call + Vente 2 Calls à strike supérieur. Réduit le coût mais crée un risque illimité au-delà d'un certain niveau. Pour une hausse modérée et précise.</SubEntry>
                <SubEntry label="Butterfly" color="#81C784">Combinaison de 3 strikes : achat K₁, vente 2×K₂, achat K₃. Pari que le spot finira exactement à K₂. Coût faible, gain concentré, profil très ciblé.</SubEntry>
              </Entry>
            </Panel>
          </>
        );
      })()}

      {/* ═══════════════════════════════════════════════════════════ */}
      {/* AIDE À LA DÉCISION */}
      {/* ═══════════════════════════════════════════════════════════ */}
      {activeTab === "decision" && (() => {
        // Position sizing
        const maxRiskAmount = portfolio * maxRiskPct / 100;
        const nbContracts = Math.floor(maxRiskAmount / premium);
        const totalPremium = nbContracts * premium;
        const totalRiskPct = (totalPremium / portfolio * 100);

        // Expected value
        const probITM = heston.probITM;
        const avgWinPayoff = heston.payoffs.filter(p => p > 0).length > 0
          ? heston.payoffs.filter(p => p > 0).reduce((a, b) => a + b, 0) / heston.payoffs.filter(p => p > 0).length : 0;
        const EV = probITM * (avgWinPayoff - premium) + (1 - probITM) * (-premium);
        const EVpct = (EV / premium * 100);
        const kellyFraction = probITM > 0 && avgWinPayoff > 0
          ? Math.max(0, (probITM * (avgWinPayoff / premium) - (1 - probITM)) / (avgWinPayoff / premium)) : 0;
        const kellyAlloc = kellyFraction * portfolio;

        // Scoring system
        const scores = {
          ev: EVpct > 20 ? 9 : EVpct > 5 ? 7 : EVpct > -5 ? 5 : EVpct > -20 ? 3 : 1,
          probITM: probITM > 0.6 ? 9 : probITM > 0.45 ? 7 : probITM > 0.3 ? 5 : probITM > 0.15 ? 3 : 1,
          riskReward: (avgWinPayoff / premium) > 5 ? 9 : (avgWinPayoff / premium) > 3 ? 7 : (avgWinPayoff / premium) > 1.5 ? 5 : 3,
          thetaCost: Math.abs(bs.theta * 30) < premium * 0.05 ? 9 : Math.abs(bs.theta * 30) < premium * 0.1 ? 7 : Math.abs(bs.theta * 30) < premium * 0.2 ? 5 : 3,
          conviction: conviction,
          horizonMatch: horizonMatch,
          volView: volView,
        };
        const weights = { ev: 0.2, probITM: 0.15, riskReward: 0.15, thetaCost: 0.1, conviction: 0.15, horizonMatch: 0.1, volView: 0.15 };
        const totalScore = Object.keys(scores).reduce((acc, k) => acc + scores[k] * weights[k], 0);
        const scoreColor = totalScore >= 7 ? "#4CAF50" : totalScore >= 5.5 ? "#FFB74D" : totalScore >= 4 ? "#FF9800" : "#F44336";
        const scoreLabel = totalScore >= 7.5 ? "EXCELLENT" : totalScore >= 6.5 ? "FAVORABLE" : totalScore >= 5 ? "NEUTRE" : totalScore >= 3.5 ? "PRUDENCE" : "DÉCONSEILLÉ";

        // What-if
        const wiS = whatIfSpot ?? S;
        const wiV = whatIfVol ?? vol;
        const wiD = whatIfDays ?? 0;
        const wiT = Math.max(0.001, T - wiD / 365);
        const wiResult = blackScholes(wiS, K, wiT, r, wiV / 100, optType);
        const wiPnl = wiResult.price - premium;
        const wiPnlPct = (wiPnl / premium) * 100;

        // Breakeven speed
        const beSpeed = Math.abs(breakeven - S) / (T * 365);

        // Score bar component
        const ScoreBar = ({ label, score, weight, color = accent }) => (
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <div style={{ width: 130, fontSize: 10, color: "#999" }}>{label}</div>
            <div style={{ flex: 1, height: 8, background: "rgba(255,255,255,0.04)", borderRadius: 4, overflow: "hidden" }}>
              <div style={{ width: `${score / 10 * 100}%`, height: "100%", background: score >= 7 ? "#4CAF50" : score >= 5 ? "#FFB74D" : "#F44336", borderRadius: 4, transition: "width 0.3s" }} />
            </div>
            <div style={{ width: 30, fontSize: 11, fontWeight: 700, color: score >= 7 ? "#4CAF50" : score >= 5 ? "#FFB74D" : "#F44336", fontFamily: "monospace", textAlign: "right" }}>{score.toFixed(1)}</div>
            <div style={{ width: 30, fontSize: 9, color: "#555" }}>×{(weight * 100).toFixed(0)}%</div>
          </div>
        );

        return (
          <>
            {/* POSITION SIZING */}
            <Panel title="Dimensionnement de Position (Position Sizing)" number="$" accent={accent}>
              <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginBottom: 14, alignItems: "flex-end" }}>
                <InputField label="Taille du portefeuille ($)" value={portfolio} onChange={setPortfolio} step={5000} width={110} />
                <InputField label="Risque max (%)" value={maxRiskPct} onChange={setMaxRiskPct} step={0.5} suffix="%" width={55} />
                <InputField label="Objectif rendement (%)" value={targetReturn} onChange={setTargetReturn} step={10} suffix="%" width={60} />
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 14 }}>
                <div style={{ background: "rgba(10,10,15,0.7)", borderRadius: 8, padding: 14 }}>
                  <div style={{ fontSize: 10, color: "#777", letterSpacing: 0.5, marginBottom: 6 }}>MÉTHODE 1 — RISQUE FIXE</div>
                  <div style={{ fontSize: 11, color: "#bbb", lineHeight: 1.7 }}>
                    Budget risque: <b style={{ color: accent }}>{fmtPrice(maxRiskAmount)}</b> ({maxRiskPct}% de {portfolio.toLocaleString()}$)<br />
                    Prix de l'option: <b style={{ color: accent }}>{fmtPrice(premium)}</b><br />
                    <span style={{ fontSize: 18, fontWeight: 700, color: accent, fontFamily: "monospace" }}>→ {nbContracts} contrats</span><br />
                    Engagement total: {fmtPrice(totalPremium)} ({totalRiskPct.toFixed(2)}% du portefeuille)
                  </div>
                </div>

                <div style={{ background: "rgba(10,10,15,0.7)", borderRadius: 8, padding: 14 }}>
                  <div style={{ fontSize: 10, color: "#777", letterSpacing: 0.5, marginBottom: 6 }}>MÉTHODE 2 — KELLY CRITERION</div>
                  <div style={{ fontSize: 11, color: "#bbb", lineHeight: 1.7 }}>
                    Kelly fraction: <b style={{ color: kellyFraction > 0 ? "#81C784" : "#E57373" }}>{(kellyFraction * 100).toFixed(1)}%</b><br />
                    Kelly allocation: <b style={{ color: accent }}>{fmtPrice(kellyAlloc)}</b><br />
                    <span style={{ fontSize: 14, fontWeight: 700, color: accent, fontFamily: "monospace" }}>→ {Math.floor(kellyAlloc / premium)} contrats (Kelly)</span><br />
                    <span style={{ fontSize: 14, fontWeight: 700, color: "#FFB74D", fontFamily: "monospace" }}>→ {Math.floor(kellyAlloc / premium * 0.5)} contrats (½ Kelly)</span><br />
                    <span style={{ fontSize: 9, color: "#666" }}>½ Kelly est recommandé en pratique (plus conservateur)</span>
                  </div>
                </div>

                <div style={{ background: "rgba(10,10,15,0.7)", borderRadius: 8, padding: 14 }}>
                  <div style={{ fontSize: 10, color: "#777", letterSpacing: 0.5, marginBottom: 6 }}>MÉTHODE 3 — OBJECTIF DE GAIN</div>
                  <div style={{ fontSize: 11, color: "#bbb", lineHeight: 1.7 }}>
                    Objectif: +{targetReturn}% = <b style={{ color: "#81C784" }}>{fmtPrice(portfolio * targetReturn / 100)}</b><br />
                    Gain moyen ITM: <b style={{ color: accent }}>{fmtPrice(avgWinPayoff - premium)}</b>/contrat<br />
                    {avgWinPayoff > premium ? <>
                      <span style={{ fontSize: 18, fontWeight: 700, color: "#81C784", fontFamily: "monospace" }}>
                        → {Math.ceil(portfolio * targetReturn / 100 / (avgWinPayoff - premium))} contrats</span><br />
                      <span style={{ fontSize: 9, color: "#666" }}>Si l'option finit ITM (prob: {(probITM * 100).toFixed(0)}%)</span>
                    </> : <span style={{ color: "#E57373" }}>E[gain ITM] négatif — objectif non atteignable</span>}
                  </div>
                </div>
              </div>
            </Panel>

            {/* SCORING / DECISION MATRIX */}
            <Panel title="Score de Décision — Matrice Multicritère" number="★" accent={accent}>
              <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 20 }}>
                <div>
                  <div style={{ fontSize: 10, color: "#777", marginBottom: 10 }}>Critères pondérés (objectifs et subjectifs)</div>
                  <ScoreBar label="Valeur Espérée (EV)" score={scores.ev} weight={weights.ev} />
                  <ScoreBar label="Probabilité ITM" score={scores.probITM} weight={weights.probITM} />
                  <ScoreBar label="Risk/Reward" score={scores.riskReward} weight={weights.riskReward} />
                  <ScoreBar label="Coût du Theta" score={scores.thetaCost} weight={weights.thetaCost} />

                  <div style={{ margin: "12px 0 8px", borderTop: "1px solid rgba(255,255,255,0.04)", paddingTop: 10 }}>
                    <div style={{ fontSize: 10, color: accent, marginBottom: 8, letterSpacing: 1 }}>VOS INPUTS SUBJECTIFS</div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <span style={{ width: 130, fontSize: 10, color: "#999" }}>Conviction (1-10)</span>
                      <input type="range" min={1} max={10} step={1} value={conviction} onChange={e => setConviction(+e.target.value)}
                        style={{ flex: 1, accentColor: accent }} />
                      <span style={{ width: 30, fontSize: 12, color: accent, fontWeight: 700, textAlign: "right" }}>{conviction}</span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <span style={{ width: 130, fontSize: 10, color: "#999" }}>Horizon adapté (1-10)</span>
                      <input type="range" min={1} max={10} step={1} value={horizonMatch} onChange={e => setHorizonMatch(+e.target.value)}
                        style={{ flex: 1, accentColor: accent }} />
                      <span style={{ width: 30, fontSize: 12, color: accent, fontWeight: 700, textAlign: "right" }}>{horizonMatch}</span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <span style={{ width: 130, fontSize: 10, color: "#999" }}>Vue sur la vol (1-10)</span>
                      <input type="range" min={1} max={10} step={1} value={volView} onChange={e => setVolView(+e.target.value)}
                        style={{ flex: 1, accentColor: accent }} />
                      <span style={{ width: 30, fontSize: 12, color: accent, fontWeight: 700, textAlign: "right" }}>{volView}</span>
                    </div>
                  </div>

                  <ScoreBar label="Conviction" score={scores.conviction} weight={weights.conviction} />
                  <ScoreBar label="Horizon adapté" score={scores.horizonMatch} weight={weights.horizonMatch} />
                  <ScoreBar label="Vue sur la vol" score={scores.volView} weight={weights.volView} />
                </div>

                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
                  <div style={{
                    width: 160, height: 160, borderRadius: "50%",
                    background: `conic-gradient(${scoreColor} ${totalScore * 10}%, rgba(255,255,255,0.04) 0%)`,
                    display: "flex", alignItems: "center", justifyContent: "center"
                  }}>
                    <div style={{
                      width: 130, height: 130, borderRadius: "50%", background: "#0c0c14",
                      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center"
                    }}>
                      <div style={{ fontSize: 36, fontWeight: 800, color: scoreColor, fontFamily: "'JetBrains Mono', monospace" }}>
                        {totalScore.toFixed(1)}
                      </div>
                      <div style={{ fontSize: 10, color: scoreColor, fontWeight: 700, letterSpacing: 1 }}>{scoreLabel}</div>
                      <div style={{ fontSize: 9, color: "#555" }}>/ 10</div>
                    </div>
                  </div>

                  <div style={{ marginTop: 14, textAlign: "center", fontSize: 11, color: "#999", lineHeight: 1.6 }}>
                    {totalScore >= 7 ? "Les indicateurs convergent positivement. La position est bien calibrée par rapport à votre profil." :
                     totalScore >= 5.5 ? "Potentiel intéressant mais certains facteurs méritent attention. Envisagez d'ajuster la taille." :
                     totalScore >= 4 ? "Rapport risque/rendement mitigé. Considérez une structure alternative ou attendez un meilleur point d'entrée." :
                     "Plusieurs signaux d'alerte. Le trade ne semble pas optimal dans sa forme actuelle."}
                  </div>
                </div>
              </div>

              {/* Key metrics summary */}
              <div style={{ marginTop: 14, display: "flex", flexWrap: "wrap", gap: 8 }}>
                <Metric small label="Valeur Espérée" value={`${EVpct >= 0 ? "+" : ""}${EVpct.toFixed(1)}%`} color={EVpct >= 0 ? "#81C784" : "#E57373"} sub={fmtPrice(EV)} />
                <Metric small label="P(ITM)" value={`${(probITM * 100).toFixed(1)}%`} color="#64B5F6" />
                <Metric small label="Gain moyen si ITM" value={fmtPrice(avgWinPayoff)} color="#81C784" sub={`${(avgWinPayoff / premium).toFixed(1)}× la prime`} />
                <Metric small label="Theta/mois" value={fmtPrice(Math.abs(bs.theta * 30))} color="#E57373" sub={`${(Math.abs(bs.theta * 30) / premium * 100).toFixed(1)}% de la prime`} />
                <Metric small label="BE speed" value={`${fmtPrice(beSpeed)}/jour`} color="#FF8A65" sub="Mvt requis par jour pour le BE" />
              </div>
            </Panel>

            {/* WHAT-IF SIMULATOR */}
            <Panel title="Simulateur What-If — Testez vos scénarios" number="?" accent={accent}>
              <div style={{ fontSize: 11, color: "#888", marginBottom: 12 }}>Ajustez les curseurs pour voir l'impact en temps réel sur votre position.</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                <div>
                  <div style={{ marginBottom: 14 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#999", marginBottom: 4 }}>
                      <span>Spot</span>
                      <span style={{ color: accent, fontFamily: "monospace", fontWeight: 600 }}>{S >= 10 ? Math.round(wiS) : wiS.toFixed(4)} ({((wiS / S - 1) * 100).toFixed(1)}%)</span>
                    </div>
                    <input type="range" min={S * 0.8} max={S * 1.2} step={S * 0.001}
                      value={wiS} onChange={e => setWhatIfSpot(+e.target.value)}
                      style={{ width: "100%", accentColor: accent }} />
                  </div>
                  <div style={{ marginBottom: 14 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#999", marginBottom: 4 }}>
                      <span>Volatilité</span>
                      <span style={{ color: "#FFB74D", fontFamily: "monospace", fontWeight: 600 }}>{wiV.toFixed(1)}% ({(wiV - vol) >= 0 ? "+" : ""}{(wiV - vol).toFixed(1)}pp)</span>
                    </div>
                    <input type="range" min={Math.max(5, vol - 15)} max={vol + 20} step={0.5}
                      value={wiV} onChange={e => setWhatIfVol(+e.target.value)}
                      style={{ width: "100%", accentColor: "#FFB74D" }} />
                  </div>
                  <div style={{ marginBottom: 14 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#999", marginBottom: 4 }}>
                      <span>Jours écoulés</span>
                      <span style={{ color: "#E57373", fontFamily: "monospace", fontWeight: 600 }}>{wiD}j / {totalDays}j ({(wiD / totalDays * 100).toFixed(0)}% du temps écoulé)</span>
                    </div>
                    <input type="range" min={0} max={totalDays - 1} step={1}
                      value={wiD} onChange={e => setWhatIfDays(+e.target.value)}
                      style={{ width: "100%", accentColor: "#E57373" }} />
                  </div>

                  <button onClick={() => { setWhatIfSpot(S); setWhatIfVol(vol); setWhatIfDays(0); }}
                    style={{ background: "rgba(255,255,255,0.05)", color: "#888", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 5, padding: "5px 16px", fontSize: 10, cursor: "pointer" }}>
                    Réinitialiser
                  </button>
                </div>

                <div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                    <Metric small label="Nouvelle Prime" value={fmtPrice(wiResult.price)} color={accent} />
                    <Metric small label="P&L" value={`${wiPnl >= 0 ? "+" : ""}${fmtPrice(wiPnl).replace("$", "")}`}
                      color={wiPnl >= 0 ? "#81C784" : "#E57373"} sub={`${wiPnlPct >= 0 ? "+" : ""}${wiPnlPct.toFixed(1)}%`} />
                    <Metric small label="Nouveau Δ" value={wiResult.delta.toFixed(4)} color="#64B5F6" />
                    <Metric small label="Nouveau Θ" value={`${wiResult.theta.toFixed(2)}/j`} color="#E57373" />
                    <Metric small label="Nouveau Vega" value={wiResult.vega.toFixed(2)} color="#FFB74D" />
                    <Metric small label="Temps restant" value={`${Math.max(0, totalDays - wiD)}j`} color="#BA68C8" />
                  </div>

                  {/* P&L decomposition */}
                  <div style={{ marginTop: 12, background: "rgba(0,0,0,0.3)", borderRadius: 6, padding: 12, fontSize: 11, lineHeight: 1.8 }}>
                    <div style={{ color: "#999", fontSize: 10, marginBottom: 6, letterSpacing: 0.5 }}>DÉCOMPOSITION DU P&L</div>
                    {(() => {
                      const deltaEffect = bs.delta * (wiS - S);
                      const gammaEffect = 0.5 * bs.gamma * (wiS - S) ** 2;
                      const vegaEffect = bs.vega * (wiV - vol);
                      const thetaEffect = bs.theta * wiD;
                      const approxPnl = deltaEffect + gammaEffect + vegaEffect + thetaEffect;
                      return <>
                        <div style={{ display: "flex", justifyContent: "space-between", color: "#64B5F6" }}>
                          <span>Effet Delta (Δ×ΔS)</span>
                          <span style={{ fontFamily: "monospace" }}>{deltaEffect >= 0 ? "+" : ""}{fmtPrice(deltaEffect).replace("$","")}</span>
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", color: "#81C784" }}>
                          <span>Effet Gamma (½Γ×ΔS²)</span>
                          <span style={{ fontFamily: "monospace" }}>{gammaEffect >= 0 ? "+" : ""}{fmtPrice(gammaEffect).replace("$","")}</span>
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", color: "#FFB74D" }}>
                          <span>Effet Vega (ν×Δσ)</span>
                          <span style={{ fontFamily: "monospace" }}>{vegaEffect >= 0 ? "+" : ""}{fmtPrice(vegaEffect).replace("$","")}</span>
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", color: "#E57373" }}>
                          <span>Effet Theta (Θ×Δt)</span>
                          <span style={{ fontFamily: "monospace" }}>{thetaEffect >= 0 ? "+" : ""}{fmtPrice(thetaEffect).replace("$","")}</span>
                        </div>
                        <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", marginTop: 6, paddingTop: 6, display: "flex", justifyContent: "space-between", color: "#ddd", fontWeight: 700 }}>
                          <span>≈ Total (approx. Taylor)</span>
                          <span style={{ fontFamily: "monospace", color: approxPnl >= 0 ? "#81C784" : "#E57373" }}>{approxPnl >= 0 ? "+" : ""}{fmtPrice(approxPnl).replace("$","")}</span>
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", color: "#888", fontSize: 10, marginTop: 4 }}>
                          <span>P&L exact (BS recalculé)</span>
                          <span style={{ fontFamily: "monospace", color: wiPnl >= 0 ? "#81C784" : "#E57373" }}>{wiPnl >= 0 ? "+" : ""}{fmtPrice(wiPnl).replace("$","")}</span>
                        </div>
                        <div style={{ fontSize: 9, color: "#555", marginTop: 4 }}>
                          L'écart entre l'approx. Taylor et le prix exact vient des termes d'ordre supérieur et des effets croisés.
                        </div>
                      </>;
                    })()}
                  </div>
                </div>
              </div>
            </Panel>

            {/* CHECKLIST */}
            <Panel title="Checklist Pré-Trade" number="✓" accent={accent}>
              {(() => {
                const checks = [
                  { label: "Le risque max (perte de la prime) est acceptable pour mon portefeuille",
                    ok: totalPremium / portfolio < 0.05,
                    detail: `Prime totale = ${(totalRiskPct).toFixed(1)}% du portefeuille ${totalRiskPct < 5 ? "(< 5% ✓)" : "(> 5% — trop exposé ?)"}` },
                  { label: "La probabilité d'expirer ITM est raisonnable",
                    ok: probITM > 0.25,
                    detail: `P(ITM) = ${(probITM * 100).toFixed(1)}% — ${probITM > 0.4 ? "Correct" : probITM > 0.25 ? "Modéré" : "Faible, option très OTM"}` },
                  { label: "La valeur espérée est positive ou quasi-neutre",
                    ok: EVpct > -10,
                    detail: `E[P&L] = ${EVpct.toFixed(1)}% de la prime ${EVpct > 0 ? "→ EV positive ✓" : EVpct > -10 ? "→ Légèrement négatif, acceptable" : "→ EV très négative"}` },
                  { label: "Le coût du theta est gérable sur l'horizon",
                    ok: Math.abs(bs.theta * 30) < premium * 0.15,
                    detail: `Theta mensuel = ${(Math.abs(bs.theta * 30) / premium * 100).toFixed(1)}% de la prime ${Math.abs(bs.theta * 30) < premium * 0.1 ? "✓" : "— attention à l'érosion"}` },
                  { label: "Le breakeven est atteignable dans l'horizon",
                    ok: pctMove < 10,
                    detail: `Mouvement requis: ${pctMove.toFixed(2)}% ${pctMove < 5 ? "→ Facilement atteignable" : pctMove < 10 ? "→ Nécessite un bon mouvement" : "→ Très ambitieux"}` },
                  { label: "La position est bien dimensionnée (Kelly / Risk-based)",
                    ok: nbContracts >= 1 && nbContracts <= 20,
                    detail: `${nbContracts} contrats (risque fixe) — ${nbContracts === 0 ? "Prime trop chère pour votre budget risque" : nbContracts > 20 ? "Possible sur-exposition" : "Dimensionnement cohérent"}` },
                  { label: "Le levier est maîtrisé",
                    ok: (S / premium) < 100,
                    detail: `Levier = ${(S / premium).toFixed(1)}x ${(S / premium) < 50 ? "✓" : "— levier élevé, ajustez la taille"}` },
                  { label: "J'ai un plan de sortie (take profit + stop loss)",
                    ok: null,
                    detail: `Suggestion: TP à +${(targetReturn).toFixed(0)}% de la prime, SL si la prime perd ${Math.min(50, Math.round(premium * 0.5))}% de sa valeur` },
                ];
                return (
                  <div>
                    {checks.map((c, i) => (
                      <div key={i} style={{
                        display: "flex", alignItems: "flex-start", gap: 10, padding: "10px 0",
                        borderBottom: i < checks.length - 1 ? "1px solid rgba(255,255,255,0.03)" : "none"
                      }}>
                        <div style={{
                          width: 22, height: 22, borderRadius: 4, flexShrink: 0, marginTop: 1,
                          background: c.ok === null ? "rgba(180,155,80,0.15)" : c.ok ? "rgba(76,175,80,0.15)" : "rgba(244,67,54,0.12)",
                          border: `1px solid ${c.ok === null ? accent : c.ok ? "#4CAF50" : "#F44336"}44`,
                          display: "flex", alignItems: "center", justifyContent: "center",
                          fontSize: 12, color: c.ok === null ? accent : c.ok ? "#4CAF50" : "#F44336"
                        }}>{c.ok === null ? "?" : c.ok ? "✓" : "✗"}</div>
                        <div>
                          <div style={{ fontSize: 12, color: "#ccc", fontWeight: 500 }}>{c.label}</div>
                          <div style={{ fontSize: 10, color: "#777", marginTop: 2 }}>{c.detail}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                );
              })()}
            </Panel>
          </>
        );
      })()}

      <div style={{ textAlign: "center", fontSize: 8, color: "#2a2a2a", marginTop: 10, paddingBottom: 16 }}>
        Black-Scholes · Heston (Euler) · Monte Carlo · Hypothèses: vol/taux constants, pas de dividendes · Usage indicatif — Ne constitue pas un conseil en investissement
      </div>
    </div>
  );
}
