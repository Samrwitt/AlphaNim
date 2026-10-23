'use strict';
const payload = JSON.parse(document.getElementById('payload').textContent);
const games = payload.games, model = payload.model, metrics = payload.metrics;
const el = id => document.getElementById(id);
let mode = 'play', human = 'a', state = 10, turn = 'a';
let moves = [], gameIndex = 0, moveIndex = 0, winner = null;
let replayTimer = null, aiTimer = null, lastDecision = null;

function tableRow(values, latest = false) {
  const row = document.createElement('tr');
  if (latest) row.className = 'latest';
  values.forEach(value => {const cell = document.createElement('td'); cell.textContent = value; row.append(cell);});
  return row;
}

games.forEach((_, i) => {
  const option = document.createElement('option');
  option.value = i; option.textContent = `Game ${i + 1} / 300`;
  el('game').append(option);
});
for (let i = 1; i <= 10; i++) {
  const stone = document.createElement('div');
  stone.className = 'stone'; stone.textContent = i;
  stone.setAttribute('aria-hidden', 'true'); el('stones').append(stone);
}
metrics.training_history.forEach(row => el('training-rows').append(tableRow([
  row.round, row.games, row.initial_loss.toFixed(4), row.final_loss.toFixed(4)
])));
el('training-result').textContent = `After training, the network alone chose optimal moves in ${metrics.after_training.network_optimal_state_count}/10 states; with search, ${metrics.after_training.search_optimal_state_count}/10. Checked against an exact solver used only for evaluation.`;

function pause() {
  if (replayTimer !== null) clearInterval(replayTimer);
  replayTimer = null;
  el('play').textContent = 'Play';
}

function cancelTimers() {
  pause();
  if (aiTimer !== null) clearTimeout(aiTimer);
  aiTimer = null;
}

function showAnalysis() {
  let decision = lastDecision;
  if (!decision && state > 0) decision = {state, result: NimEngine.search(model, state)};
  el('search-rows').replaceChildren();
  if (!decision) {
    el('analysis-context').textContent = 'No further decision: the game has ended.';
    el('value-info').textContent = '';
    return;
  }
  const result = decision.result;
  el('analysis-context').textContent = lastDecision
    ? `AlphaZero’s last decision: ${decision.state} stones, chose to remove ${result.action}.`
    : mode === 'watch'
      ? `Final model’s analysis of the current ${decision.state}-stone position. This is not the historical training policy.`
      : `Final model’s analysis of the current ${decision.state}-stone position.`;
  for (let i = 0; i < Math.min(2, decision.state); i++) {
    el('search-rows').append(tableRow([`Take ${i + 1}`, `${(result.prior[i] * 100).toFixed(1)}%`,
      result.visits[i], `${(result.policy[i] * 100).toFixed(1)}%`, result.q[i].toFixed(3)], i + 1 === result.action));
  }
  el('value-info').textContent = `Network’s expected signed result: ${result.value.toFixed(3)} (−1 = loss, +1 = win, from the player-to-move perspective). This is an estimate, not a guaranteed outcome.`;
}

function render() {
  const finished = state === 0, live = mode === 'play';
  el('remaining').textContent = state;
  el('stones').setAttribute('aria-label', `${state} stones remaining`);
  [...el('stones').children].forEach((stone, i) => stone.classList.toggle('removed', i >= state));
  for (const player of ['a', 'b']) {
    const name = live ? (player === human ? `You · ${player.toUpperCase()}` : `AlphaZero · ${player.toUpperCase()}`) : `Player ${player.toUpperCase()}`;
    el(player).textContent = `${name}${finished && player === winner ? ' · Winner!' : !finished && turn === player ? ' · To move' : ''}`;
    el(player).classList.toggle('active', finished ? player === winner : player === turn);
  }
  el('status').textContent = finished
    ? (live ? (winner === human ? 'You took the last stone. You win!' : 'AlphaZero took the last stone. AlphaZero wins!') : `Player ${winner.toUpperCase()} took the last stone and wins!`)
    : live ? (turn === human ? 'Your turn. Take 1 or 2 stones.' : 'AlphaZero is thinking…') : `Player ${turn.toUpperCase()} goes next. Press Play or Next move.`;
  el('progress').textContent = live ? `Move ${moves.length} · You are ${human.toUpperCase()}`
    : `Training game ${gameIndex + 1} / 300 · Round ${Math.floor(gameIndex / 30) + 1} · Move ${moveIndex} / ${games[gameIndex].length}`;
  el('turn-info').textContent = finished ? `Final state: 0 stones · Winner: ${winner.toUpperCase()} · All outcomes are now shown.`
    : `Current state: ${state} stones · Next player: ${turn.toUpperCase()}`;
  el('seat-controls').hidden = !live; el('human-controls').hidden = !live;
  el('game-controls').hidden = live; el('replay-controls').hidden = live;
  el('reset').textContent = live ? 'New game' : 'Restart';
  el('take-one').disabled = !live || finished || turn !== human;
  el('take-two').disabled = !live || finished || turn !== human || state < 2;
  el('step').disabled = finished; el('play').disabled = finished;
  el('rows').replaceChildren();
  if (!moves.length) {
    const row = document.createElement('tr'), cell = document.createElement('td');
    cell.colSpan = 6; cell.className = 'empty';
    cell.textContent = live ? 'Your moves and AlphaZero’s moves will appear here.' : 'Press Play or Next move to see the training game.';
    row.append(cell); el('rows').append(row);
  }
  moves.forEach(([before, action, outcome, player], i) => {
    const label = finished ? (player === winner ? '+1' : '−1') : 'Pending';
    el('rows').append(tableRow([i + 1, before, action, before - action, label, player], i === moves.length - 1));
  });
  showAnalysis();
}

function applyMove(action) {
  if (state === 0 || !Number.isInteger(action) || action < 1 || action > Math.min(2, state)) return false;
  moves.push([state, action, null, turn]);
  state -= action;
  if (state === 0) {
    winner = turn;
    moves = moves.map(([s, a, z, player]) => [s, a, player === winner ? 1 : -1, player]);
  } else turn = turn === 'a' ? 'b' : 'a';
  return true;
}

function scheduleAI() {
  if (mode !== 'play' || state === 0 || turn === human || aiTimer !== null) return;
  aiTimer = setTimeout(() => {
    aiTimer = null;
    if (mode !== 'play' || state === 0 || turn === human) return;
    const before = state, result = NimEngine.search(model, state);
    lastDecision = {state: before, result};
    applyMove(result.action); render();
  }, 500);
}

function humanMove(action) {
  if (mode !== 'play' || state === 0 || turn !== human) return;
  if (!applyMove(action)) return;
  lastDecision = null;
  render(); scheduleAI();
}

function reset() {
  cancelTimers();
  state = 10; turn = 'a'; moves = []; moveIndex = 0; winner = null; lastDecision = null;
  render(); scheduleAI();
}

function replayNext() {
  if (mode !== 'watch' || state === 0) return;
  const row = games[gameIndex][moveIndex++];
  applyMove(row[1]);
  if (state === 0) pause();
  render();
}

function startReplay() {
  if (mode !== 'watch' || state === 0 || replayTimer !== null) return;
  replayTimer = setInterval(replayNext, Number(el('speed').value));
  el('play').textContent = 'Pause';
}

el('take-one').addEventListener('click', () => humanMove(1));
el('take-two').addEventListener('click', () => humanMove(2));
el('mode').addEventListener('change', () => {mode = el('mode').value; reset();});
el('seat').addEventListener('change', () => {human = el('seat').value; reset();});
el('game').addEventListener('change', () => {gameIndex = Number(el('game').value); reset();});
el('reset').addEventListener('click', reset);
el('play').addEventListener('click', () => {if (replayTimer === null) startReplay(); else pause();});
el('step').addEventListener('click', () => {pause(); replayNext();});
el('speed').addEventListener('change', () => {if (replayTimer !== null) {pause(); startReplay();}});
reset();
