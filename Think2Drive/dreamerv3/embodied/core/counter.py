import functools
import threading
import multiprocessing as mp


@functools.total_ordering
class Counter:

  def __init__(self, initial=0):
    self.value = initial
    self.lock = threading.Lock()

  def __repr__(self):
    return f'Counter({self.value})'

  def __int__(self):
    return int(self.value)

  def __eq__(self, other):
    return int(self) == other

  def __ne__(self, other):
    return int(self) != other

  def __lt__(self, other):
    return int(self) < other

  def __add__(self, other):
    return int(self) + other

  def __radd__(self, other):
    return other - int(self)

  def __sub__(self, other):
    return int(self) - other

  def __rsub__(self, other):
    return other - int(self)

  def increment(self, amount=1):
    with self.lock:
      self.value += amount

  def reset(self):
    with self.lock:
      self.value = 0

  def save(self):
    return self.value

  def load(self, value):
    self.value = value

@functools.total_ordering
class MPCounter:
  """
  A thread-safe counter class with support for basic arithmetic operations.

  This class uses the @total_ordering decorator, which generates all ordering 
  operations based on the provided __eq__ and __lt__ methods.
  """

  def __init__(self, initial=0):
    """
    Initialize the Counter.

    Args:
        initial (int): The initial value of the counter. Defaults to 0.
    """
    manager = mp.Manager()
    self.value = manager.Value(int, initial)
    self.lock = manager.Lock()

  def __repr__(self):
    """Return a string representation of the Counter."""
    return f'Counter({self.value.value})'

  def __int__(self):
    """Allow the Counter to be used as an integer."""
    return int(self.value.value)

  def __eq__(self, other):
    """Check if the Counter is equal to another value."""
    return int(self) == other

  def __ne__(self, other):
    """Check if the Counter is not equal to another value."""
    return int(self) != other

  def __lt__(self, other):
    """Check if the Counter is less than another value."""
    return int(self) < other

  def __add__(self, other):
    """Add another value to the Counter."""
    return int(self) + other

  def __radd__(self, other):
    """Support right-side addition."""
    return other - int(self)

  def __sub__(self, other):
    """Subtract another value from the Counter."""
    return int(self) - other

  def __rsub__(self, other):
    """Support right-side subtraction."""
    return other - int(self)

  def increment(self, amount=1):
    """
    Increment the counter in a thread-safe manner.

    Args:
        amount (int): The amount to increment by. Defaults to 1.
    """
    with self.lock:
        value = self.value.value

        self.value.value += amount

        return value

  def reset(self):
    """Reset the counter to zero in a thread-safe manner."""
    with self.lock:
        self.value.value = 0

  def save(self):
    """Return the current value of the counter."""
    return self.value.value

  def load(self, value):
    """
    Set the counter to a specific value.

    Args:
        value: The value to set the counter to.
    """
    self.value.value = value


@functools.total_ordering
class EveryNCounter:
  """
  A counter that triggers every N increments, with basic arithmetic support.
  """

  def __init__(self, n, initial=0):
    """
    Initialize the EveryNCounter.

    Args:
        n (int): Number of increments to trigger the counter.
        initial (int): Initial counter value. Defaults to 0.
    """
    self.value = initial
    self.n = n

  def __repr__(self):
    return f'EveryNCounter(value={self.value}, n={self.n})'

  def __int__(self):
    return int(self.value)

  def __eq__(self, other):
    return int(self) == other

  def __lt__(self, other):
    return int(self) < other

  def __add__(self, other):
    return int(self) + other

  def __radd__(self, other):
    return other + int(self)

  def __sub__(self, other):
    return int(self) - other

  def __rsub__(self, other):
    return other - int(self)

  def increment(self, amount=1):
    """
    Increment the counter.

    Args:
        amount (int): Increment amount. Defaults to 1.

    Returns:
        bool: True if counter reached a multiple of N, False otherwise.
    """
    self.value += amount
    return self.value % self.n == 0

  def reset(self):
    """Reset the counter to zero."""
    self.value = 0

  def save(self):
    """Return the current counter value."""
    return self.value

  def load(self, value):
    """Set the counter to a specific value."""
    self.value = value

  def __call__(self, steps=1):
    """
    Increment counter and check if it triggers.

    Args:
        steps (int): Number of steps to increment. Defaults to 1.

    Returns:
        bool: True if counter reached a multiple of N, False otherwise.
    """
    return self.increment(steps)