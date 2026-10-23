/* The trained NumPy network and deterministic PUCT search, ported to JavaScript.
   No lookup-table strategy: both predictions and search execute in the browser. */
'use strict';
const NimEngine = (() => {
  function predict(model, state) {
    if (!Number.isInteger(state) || state < 1 || state > 10) throw new Error('State must be 1–10.');
    const h = model.w[state - 1].map((w, i) => Math.tanh(w + model.b[i]));
    const logits = model.bp.map((bias, a) => h.reduce((sum, v, i) => sum + v * model.wp[i][a], bias));
    if (state === 1) logits[1] = -1e9;
    const max = Math.max(...logits);
    const exp = logits.map(v => Math.exp(v - max));
    const total = exp[0] + exp[1];
    const value = Math.tanh(h.reduce((sum, v, i) => sum + v * model.wv[i][0], model.bv[0]));
    return {policy: exp.map(v => v / total), value};
  }

  function search(model, state, simulations = 64) {
    if (!Number.isInteger(simulations) || simulations < 1) throw new Error('Search needs a positive simulation count.');
    function node(s) {
      const prediction = predict(model, s);
      return {state: s, prior: prediction.policy, value: prediction.value,
        visits: [0, 0], totals: [0, 0], children: {}};
    }
    function simulate(current) {
      const total = current.visits[0] + current.visits[1];
      let edge = 0, best = -Infinity;
      for (let a = 0; a < Math.min(2, current.state); a++) {
        const q = current.totals[a] / Math.max(1, current.visits[a]);
        const u = 1.5 * current.prior[a] * Math.sqrt(1 + total) / (1 + current.visits[a]);
        if (q + u > best) {best = q + u; edge = a;}
      }
      const remaining = current.state - edge - 1;
      let result;
      if (remaining === 0) result = 1;
      else if (!(edge in current.children)) {
        const child = node(remaining);
        current.children[edge] = child;
        result = -child.value;
      } else result = -simulate(current.children[edge]);
      current.visits[edge]++;
      current.totals[edge] += result;
      return result;
    }
    const root = node(state);
    for (let i = 0; i < simulations; i++) simulate(root);
    return {action: root.visits[1] > root.visits[0] ? 2 : 1,
      policy: root.visits.map(n => n / simulations), visits: root.visits,
      q: root.totals.map((v, i) => v / Math.max(1, root.visits[i])),
      prior: root.prior, value: root.value};
  }
  return {predict, search};
})();
if (typeof module !== 'undefined') module.exports = NimEngine;
