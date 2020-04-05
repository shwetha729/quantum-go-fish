"""Classes and methods which implement game logic."""

import collections
from typing import Dict, Iterator, List, Tuple, Union
import numpy as np


PLAYER_COUNT_BITS = 3  # Enough bits to encode the number of players
CARDS_PER_SUIT = 4

Ask = collections.namedtuple('Ask', ['player', 'suit'])
Answer = collections.namedtuple('Answer', ['give_card'])
Move = Union[Ask, Answer, None]

class Position:
  """A position in the game.

  Attributes:
  * curr_player (assumed 0)
  * active_player (assumed equal to curr_player)
  * requested_suit (assumed 0)
  * known (inferred from states or packed)
  * unknown (inferred from states or packed)
  * is_excluded (inferred from states or packed)
  * states (inferred from packed or known, unknown, and is_excluded)
  * packed (inferred from everything else)
  """
  def __init__(self, **kwargs):
    if 'packed' in kwargs:
      kwargs = unpack_position(kwargs['packed'])
    has_states = ('states' in kwargs)
    has_k_u_x = all(x in kwargs for x in ('known', 'unknown', 'is_excluded'))
    if not (has_states or has_k_u_x):
      raise ValueError('Either packed, states, or (known, unknown, is_excluded)'
                       ' must be provided')
    for k, v in kwargs.items():
      # Set states last so that it overwrites known, unknown, is_excluded.
      if k != 'states':
        setattr(self, k, v)
    if has_states:
      self.states = kwargs['states']

  def __getattr__(self, attr):
    # Lazy initialization of inferrable attributes.
    if attr in ('known', 'unknown', 'is_excluded'):
      k, u, x = states_to_known_unknown_excluded(self.states)
      self.known, self.unknown, self.is_excluded = k, u, x
    elif attr == 'curr_player':
      self.curr_player = 0
    elif attr == 'active_player':
      self.active_player = self.curr_player
    elif attr == 'requested_suit':
      self.requested_suit = 0
    elif attr == 'packed':
      self.packed = pack_position(self)
    elif attr == 'states':
      self.states = known_unknown_excluded_to_states(
          self.known, self.unknown, self.is_excluded)
    else:
      raise AttributeError(f"'{type(self).__name__}' object has no attribute "
                           f"'{attr}'")
    return getattr(self, attr)

  @property
  def states(self):
    return self._states

  @states.setter
  def states(self, states):
    self._states = states
    # Reset attributes which should be inferred from new states.
    for attr in ('known', 'unknown', 'is_excluded', 'packed'):
      if hasattr(self, attr):
        delattr(self, attr)

  def __str__(self):
    strs = [f'   known   unknown  ({len(self.states)} possibilities) '
            f'{self.packed}']
    is_option = ((self.states - self.known).max(axis=0) > 0)
    for player in range(self.num_players):
      s = str(player) + ": "
      ops = ''
      for suit in range(self.num_players):
        s += str(suit) * self.known[suit, player]
        if is_option[suit, player]:
          ops += str(suit)
      s = s.ljust(11)
      u = '?' * self.unknown[player]
      if u:
        s += u + '{' + ops + '}'
      s = s.ljust(22)
      if player == self.curr_player:
        s += '< current'
      elif player == self.active_player:
        s += f'< got a {self.requested_suit}?'
      strs.append(s)
    return '\n'.join(strs)

  @property
  def num_players(self):
    """Number of players."""
    return self.known.shape[0]


def legal_moves(position: Position) -> List[Move]:
  """Returns a list of legal moves from the given positions."""
  me = position.active_player
  if me != position.curr_player:
    # I'm answering a question.
    legal_responses = []
    if not position.is_excluded[position.requested_suit, me]:
      legal_responses.append(Answer(True))
    if not position.known[position.requested_suit, me]:
      legal_responses.append(Answer(False))
    return legal_responses
  # I get to ask a question.
  moves = []
  for suit in range(position.num_players):
    if not position.is_excluded[suit, me]:
      for p in range(position.num_players):
        if p == me:
          continue
        # Prohibit asking if you know the answer is no
        if not position.is_excluded[suit, p]:
          moves.append(Ask(p, suit))
  if moves:
    return moves
  return [None]

def do_move(pos: Position, move: Move) -> Position:
  """Returns the position after doing `move` from position `pos`."""
  pos_kwargs = dict()
  next_player = (pos.curr_player + 1) % pos.num_players
  if move is None:
    # player without any cards passes.
    pos_kwargs.update(curr_player=next_player, states=pos.states)
  elif isinstance(move, Ask):
    # prune to states where current player least one card of the requested suit.
    states = pos.states[pos.states[:, move.suit, pos.curr_player] > 0]
    pos_kwargs.update(curr_player=pos.curr_player,
                      active_player=move.player,
                      requested_suit=move.suit,
                      states=states)
  elif move.give_card:
    # prune to cases where active player has the requested suit
    states = pos.states[pos.states[:, pos.requested_suit, pos.active_player] > 0]
    # transfer a card
    states[:, pos.requested_suit, pos.active_player] -= 1
    states[:, pos.requested_suit, pos.curr_player] += 1
    pos_kwargs.update(curr_player=next_player, states=states)
  else:
    # prune to cases where active player has none of the requested suit
    states = pos.states[pos.states[:, pos.requested_suit, pos.active_player] == 0]
    pos_kwargs.update(curr_player=next_player, states=states)
  return Position(**pos_kwargs)


def winner(pos: Position) -> Union[int, None]:
  """Returns the winner in a given position, or None.

  If there is a winner, this assumes we *just* got to a winning state."""
  if len(pos.states) == 1:
    if pos.active_player != pos.curr_player:
      # current player won by asking a question
      return pos.curr_player
    # previous player won after getting a response
    return (pos.curr_player - 1) % pos.num_players
  has_all_of_a_kind = (pos.states == CARDS_PER_SUIT).any(axis=1).all(axis=0)
  if has_all_of_a_kind.any():
    # Of players who have all cards of a kind, whichever one would play next
    # wins.
    p = pos.curr_player
    while not has_all_of_a_kind[p]:
      p = (p + 1) % pos.num_players
    return p
  return None


################### Methods below here are "internal-only". ###################

def consistent_distributions(num_cards: int,
                             upper_bound: List[int],
                             distribution_so_far: Tuple[int, ...] = tuple(),
                             ) -> Iterator[Tuple[int, ...]]:
  """Iterator over all distributions of `num_cards` identical cards among
  players, subject to the constraint that player p not get more than
  `upper_bound[p]` cards.

  Arguments:
    * num_cards: Int, the number of cards to distribute.
    * upper_bound: List[Int], the p-th entry is the maximum number player p can
      be assigned.

  Yields:
    Tuples where the p-the entry is the number of cards player p gets.
  """

  curr_player = len(distribution_so_far)
  if sum(upper_bound[curr_player:]) < num_cards:
    # There are too many cards to distribute given the remaining upper bounds.
    return

  if len(upper_bound) == curr_player:
    # We've distributed all the cards to the players.
    yield distribution_so_far
  else:
    min_cards_to_curr_player = max(
        0, num_cards - sum(upper_bound[curr_player + 1:]))
    max_cards_to_curr_player = min(num_cards, upper_bound[curr_player])
    for i in range(min_cards_to_curr_player, max_cards_to_curr_player + 1):
      # Assign `i` cards to the current player.
      for distribution in consistent_distributions(
          num_cards - i, upper_bound, distribution_so_far + (i,)):
        yield distribution


def consistent_states(suit_to_num_cards: np.ndarray,
                      player_to_num_cards: np.ndarray,
                      is_excluded: np.ndarray,
                      suits_distributed_so_far: Tuple[Tuple[int, ...], ...] = tuple()):
  """Iterator over all states consistent with the given information.

  Arguments:
    * suit_to_num_cards: np.ndarray of ints, the s-th entry is the number of
      outstanding cards of suit s.
    * player_to_num_cards: np.ndarray of ints, the p-the entry is the number of
      unknown cards held by player p.
    * is_excluded: np.ndarray of bools, the (s, p)-th entry is whether player p
      is guaranteed to have zero cards of suit s.

  Yields:
    2-dimensional numpy arrays, where the (s, p)-th entry is the number of cards
    of suit s in player p's hand.
  """

  # assert suit_to_num_cards.sum() == player_to_num_cards.sum()
  suit = len(suits_distributed_so_far)
  if len(suit_to_num_cards) == suit:
    # All suits have been distributed.
    yield np.array(suits_distributed_so_far)
  else:
    upper_bound = player_to_num_cards * (~is_excluded[suit])
    for suit_distribution in consistent_distributions(suit_to_num_cards[suit], upper_bound):
      # Distribute this suit as given, and iterate over all ways to distribute
      # the remaining suits.
      for state in consistent_states(
          suit_to_num_cards, player_to_num_cards - suit_distribution,
          is_excluded,
          suits_distributed_so_far + (suit_distribution,)):
        yield state

def all_possible_initial_states(num_players: int) -> np.ndarray:
  """All initial states for a game with the given number of players."""
  is_excluded = np.zeros((num_players, num_players)).astype(bool)
  suit_to_num_cards = CARDS_PER_SUIT * np.ones(num_players, dtype=int)
  player_to_num_cards = CARDS_PER_SUIT * np.ones(num_players, dtype=int)
  return np.array(list(consistent_states(suit_to_num_cards, player_to_num_cards, is_excluded)))

def states_to_known_unknown_excluded(
    states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns known, unknown, and excluded cards given an array of states."""
  is_excluded = states.max(axis=0) == 0
  known_cards = states.min(axis=0)
  num_cards_held = states[0].sum(axis=0)
  num_unknown_cards = num_cards_held - known_cards.sum(axis=0)
  return known_cards, num_unknown_cards, is_excluded

def known_unknown_excluded_to_states(
    known: np.ndarray, unknown: np.ndarray, is_excluded: np.ndarray) -> np.ndarray:
  """Returns possible states given known, unknown. and excluded cards."""
  suit_to_num_cards = CARDS_PER_SUIT - known.sum(axis=1)
  return np.array(list(consistent_states(suit_to_num_cards, unknown, is_excluded))) + known


def pack_array(arr: np.ndarray, bits_per_entry: int, n: int) -> int:
  """Packs an integer array into the low order bits."""
  for x in arr.ravel():
    n = (n << bits_per_entry) + int(x)
  return n

# For some reason pytype is unhappy with return type Tuple[np.ndarray, int].
def unpack_array(shape: Tuple[int, ...], bits_per_entry: int, n: int) -> Tuple:
  """Unpacks an integer array from the low order bits of n."""
  arr = []
  for _ in range(np.prod(shape)):
    n, x = divmod(n, 2**bits_per_entry)
    arr.append(x)

  return np.array(arr[::-1]).reshape(shape), n

def bit_counts(num_players: int) -> Tuple[int, int, int]:
  """Returns numbers of bits needed for packing/unpacking positions.

  * bits_player is enough bits to encode a player identity, [0, num_players),
  * bits_cards is enough bits to encode a number of cards of a given suit,
    [0, CARDS_PER_SUIT],
  * bits_suit is enough bits to encode a suit, [0, num_suits).
  """
  bits_player = int(np.ceil(np.log2(num_players)))
  bits_cards = int(np.ceil(np.log2(CARDS_PER_SUIT + 1)))
  bits_suit = bits_player  # same number of players as suits
  return bits_player, bits_cards, bits_suit

def pack_position(pos: Position) -> int:
  """Returns an integer encoding the given position."""
  bits_player, bits_cards, bits_suit = bit_counts(pos.num_players)
  n = pos.curr_player
  n = (n << bits_player) + pos.active_player
  n = (n << bits_suit) + pos.requested_suit
  n = pack_array(pos.known, bits_cards, n)
  n = pack_array(pos.unknown, bits_cards, n)
  n = pack_array(pos.is_excluded, 1, n)
  n = (n << PLAYER_COUNT_BITS) + pos.num_players
  return n

def unpack_position(n: int) -> Dict:
  """Returns kwargs for position with given integer encoding."""
  kwargs = dict(packed=n)
  n, num_players = divmod(n, 2**PLAYER_COUNT_BITS)
  bits_player, bits_cards, bits_suit = bit_counts(num_players)
  is_excluded, n = unpack_array((num_players, num_players), 1, n)
  is_excluded = is_excluded.astype(bool)
  unknown, n = unpack_array((num_players,), bits_cards, n)
  known, n = unpack_array((num_players, num_players), bits_cards, n)
  n, requested_suit = divmod(n, 2**bits_suit)
  n, active_player = divmod(n, 2**bits_player)
  n, curr_player = divmod(n, 2**bits_player)
  kwargs.update(curr_player=curr_player, active_player=active_player,
      requested_suit=requested_suit, known=known, unknown=unknown,
      is_excluded=is_excluded)
  return kwargs

