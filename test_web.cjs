/* Run with node test_web.cjs. Checks static JavaScript without a browser server. */
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const engine = require('./engine.js');
const model = JSON.parse(fs.readFileSync('artifacts/model.json', 'utf8'));
const games = JSON.parse(fs.readFileSync('data/self_play_games.json', 'utf8'));
const metrics = JSON.parse(fs.readFileSync('artifacts/alphazero_metrics.json', 'utf8'));
const expected = metrics.after_training.positions;
for (let s = 1; s <= 10; s++) {
  const p = engine.predict(model, s), result = engine.search(model, s);
  assert(Math.abs(p.value - expected[s - 1].value) < 1e-12);
  p.policy.forEach((value, i) => assert(Math.abs(value - expected[s - 1].policy[i]) < 1e-12));
  assert.deepEqual(result.policy, expected[s - 1].search_policy);
  assert.equal(result.visits[0] + result.visits[1], 64);
}
assert.throws(() => engine.predict(model, 0));
assert.equal(engine.search(model, 1).action, 1);
console.log('PASS: browser network predictions and MCTS match Python in all 10 states.');

class Element {
  constructor() {this.children = []; this.handlers = {}; this.value = '1000'; this.textContent = ''; this.classList = {toggle() {}};}
  append(child) {this.children.push(child);}
  replaceChildren() {this.children = [];}
  setAttribute() {}
  addEventListener(name, callback) {this.handlers[name] = callback;}
}
function createApp() {
  const elements = {}, callbacks = new Map(); let nextTimer = 1;
  const get = id => elements[id] ||= new Element();
  get('payload').textContent = JSON.stringify({games, model, metrics});
  const context = vm.createContext({document: {getElementById: get, createElement: () => new Element()},
    setTimeout(fn) {const id = nextTimer++; callbacks.set(id, fn); return id;},
    clearTimeout(id) {callbacks.delete(id);},
    setInterval(fn) {const id = nextTimer++; callbacks.set(id, fn); return id;},
    clearInterval(id) {callbacks.delete(id);}});
  vm.runInContext(fs.readFileSync('engine.js', 'utf8') + '\n' + fs.readFileSync('app.js', 'utf8'), context);
  return {get, run: code => vm.runInContext(code, context),
    tick() {for (const [id, fn] of [...callbacks]) {callbacks.delete(id); fn();}}, callbacks};
}
const app = createApp();
app.get('mode').value = 'watch'; app.get('mode').handlers.change();
assert.equal(app.get('game').children.length, 300);
for (let i = 0; i < 300; i++) {
  app.get('game').value = String(i); app.get('game').handlers.change();
  for (let j = 0; j < games[i].length; j++) app.get('step').handlers.click();
  assert.equal(Number(app.get('remaining').textContent), 0);
  assert.equal(app.get('rows').children.length, games[i].length);
  app.get('rows').children.forEach((row, j) => {
    assert.equal(row.children[1].textContent, games[i][j][0]);
    assert.equal(row.children[2].textContent, games[i][j][1]);
    assert.equal(row.children[4].textContent, games[i][j][2] === 1 ? '+1' : '−1');
  });
  assert(app.get('turn-info').textContent.includes('Final state: 0'));
}
console.log('PASS: all 300 training replays, final states and outcome tables.');

let completed = 0;
function explore(human, prefix = []) {
  const app = createApp();
  app.get('seat').value = human; app.get('seat').handlers.change(); app.tick();
  for (const action of prefix) {
    app.get(action === 1 ? 'take-one' : 'take-two').handlers.click(); app.tick();
  }
  const state = app.run('state');
  if (state === 0) {
    completed++;
    assert(app.get('status').textContent.includes('win'));
    assert(app.get('take-one').disabled && app.get('take-two').disabled);
    const length = app.run('moves.length');
    app.get('take-one').handlers.click(); assert.equal(app.run('moves.length'), length);
    return;
  }
  assert.equal(app.run('turn'), human);
  assert(!app.get('take-one').disabled);
  if (state === 1) assert(app.get('take-two').disabled);
  for (let action = 1; action <= Math.min(2, state); action++) explore(human, [...prefix, action]);
}
explore('a'); explore('b');
// Changing mode or restarting during thinking must cancel the old AI turn.
const race = createApp();
race.get('take-one').handlers.click();
assert.equal(race.run('state'), 9);
race.get('take-one').handlers.click(); assert.equal(race.run('state'), 9);
race.get('mode').value = 'watch'; race.get('mode').handlers.change(); race.tick();
assert.equal(race.run('state'), 10);
race.get('reset').handlers.click(); assert.equal(race.run('moves.length'), 0);
console.log(`PASS: ${completed} complete human-vs-agent paths across both seats; turn locking, illegal moves, reset and timer cancellation.`);
