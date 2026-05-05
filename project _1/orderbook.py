"""
============================================================
  ORDER BOOK SIMULATOR — Complexity Analysis
============================================================

Q1: Why is a heap (priority queue) the right data structure for an order book?
    What would happen if you used a sorted list instead?

    A heap gives us O(log n) insertion and O(1) access to the best
    (highest bid or lowest ask) element at all times — exactly the
    two operations an order book does most. A sorted list would also
    give O(1) access to the best element, but *inserting* a new order
    requires O(n) time because every element after the insertion point
    must shift. At high throughput (thousands of orders/second) that
    linear insertion cost becomes a bottleneck, while the heap keeps
    every insertion cheap regardless of book depth.

Q2: What is the time complexity of match_orders() in the worst case?
    Show your reasoning step by step.

    Let k = number of trades executed in one call.
    Each trade requires:
      - O(1)      peek at best bid and best ask
      - O(log n)  heappop from the bid side
      - O(log n)  heappop from the ask side
      - O(log n)  optional heappush (partial fill remainder)
    So each trade costs O(log n). With k trades the total is O(k log n).
    In the absolute worst case every order matches every other order,
    so k can be up to min(|bids|, |asks|) ≈ n/2, giving O(n log n).

Q3: If n = 1,000,000 active orders, how many operations does
    match_orders() perform in the worst case? Is this acceptable?

    Worst case: k ≈ 500,000 trades, each costing ~20 heap operations
    (log₂ 1,000,000 ≈ 20). Total ≈ 10,000,000 operations.
    Modern CPUs execute ~10⁸–10⁹ simple operations per second, so
    this completes in roughly 10–100 ms — acceptable for a batch
    matching run. Real exchanges process orders one at a time
    (O(log n) per event) to keep per-event latency in microseconds.

Q4: Compare heap complexity to a naive plain-list implementation.
    At what value of n does the heap become meaningfully faster?

    A plain-list approach must scan all orders to find the best bid/ask:
    O(n) per lookup. match_orders() then becomes O(n²) in the worst
    case (each of the n/2 trades triggers another O(n) scan).
    The heap version is O(n log n). They cross over around n ≈ 10–20:
    for small books the constant factors dominate and lists are fine,
    but beyond ~50 orders the heap pulls decisively ahead. At
    n = 10,000 the naive approach does ~10⁸ operations vs ~10⁵ for
    the heap — three orders of magnitude faster.

============================================================
"""

import heapq
import time
import random


# ---------------------------------------------------------------------------
# OrderBook
# ---------------------------------------------------------------------------

class OrderBook:
    """
    A simplified limit order book backed by two heaps.

    Internal representation
    -----------------------
    self._bids : min-heap of (-price, quantity)
        Negating the price turns heapq (a min-heap) into a max-heap so
        that heapq.heappop always returns the *highest* bid.

    self._asks : min-heap of (price, quantity)
        Standard min-heap; heapq.heappop returns the *lowest* ask.
    """

    def __init__(self):
        self._bids: list[tuple[float, int]] = []   # (-price, quantity)
        self._asks: list[tuple[float, int]] = []   # ( price, quantity)

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def place_bid(self, price: float, quantity: int) -> None:
        """Add a buy order at the given price and quantity.  O(log n)."""
        if quantity <= 0:
            raise ValueError("Quantity must be positive.")
        heapq.heappush(self._bids, (-price, quantity))

    def place_ask(self, price: float, quantity: int) -> None:
        """Add a sell order at the given price and quantity.  O(log n)."""
        if quantity <= 0:
            raise ValueError("Quantity must be positive.")
        heapq.heappush(self._asks, (price, quantity))

    # ------------------------------------------------------------------
    # Best prices   (O(1) — heap root is always the best element)
    # ------------------------------------------------------------------

    def best_bid(self) -> float | None:
        """Return the highest bid price, or None if no bids exist."""
        if not self._bids:
            return None
        return -self._bids[0][0]   # undo negation

    def best_ask(self) -> float | None:
        """Return the lowest ask price, or None if no asks exist."""
        if not self._asks:
            return None
        return self._asks[0][0]

    def spread(self) -> float | None:
        """Return best_ask − best_bid, or None if either side is empty."""
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return round(ba - bb, 10)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match_orders(self) -> list[dict]:
        """
        Execute trades whenever best_bid >= best_ask.

        Rules
        -----
        - Trade price    = ask price (seller sets the floor)
        - Trade quantity = min(bid_qty, ask_qty)
        - Partial fills  : remainder stays in the book
        - Continues until no matchable pair remains

        Returns
        -------
        List of trade dicts:
          {'price': float, 'quantity': int,
           'buyer_paid': float, 'seller_received': float}
        """
        trades: list[dict] = []

        while self._bids and self._asks:
            best_bid_price = -self._bids[0][0]
            best_ask_price =  self._asks[0][0]

            if best_bid_price < best_ask_price:
                break   # no match possible

            # Pop both sides
            neg_bid_price, bid_qty = heapq.heappop(self._bids)
            ask_price,     ask_qty = heapq.heappop(self._asks)

            trade_price = ask_price
            trade_qty   = min(bid_qty, ask_qty)

            trades.append({
                'price':           round(trade_price, 10),
                'quantity':        trade_qty,
                'buyer_paid':      round(trade_price * trade_qty, 10),
                'seller_received': round(trade_price * trade_qty, 10),
            })

            # Handle partial fills — push remainder back
            remaining_bid = bid_qty - trade_qty
            remaining_ask = ask_qty - trade_qty

            if remaining_bid > 0:
                heapq.heappush(self._bids, (neg_bid_price, remaining_bid))
            if remaining_ask > 0:
                heapq.heappush(self._asks, (ask_price, remaining_ask))

        return trades

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        bb = self.best_bid()
        ba = self.best_ask()
        sp = self.spread()
        return (
            f"OrderBook(bids={len(self._bids)}, asks={len(self._asks)}, "
            f"best_bid={bb}, best_ask={ba}, spread={sp})"
        )


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def run_tests():
    print("=" * 55)
    print("Running unit tests …")
    print("=" * 55)
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))
            failed += 1

    # ---------------------------------------------------------------
    # Test 1: Empty book
    # ---------------------------------------------------------------
    book = OrderBook()
    check("empty book: best_bid is None",  book.best_bid() is None)
    check("empty book: best_ask is None",  book.best_ask() is None)
    check("empty book: spread is None",    book.spread()   is None)
    check("empty book: match returns []",  book.match_orders() == [])

    # ---------------------------------------------------------------
    # Test 2: Single match, exact quantities
    # ---------------------------------------------------------------
    book = OrderBook()
    book.place_ask(price=0.50, quantity=10)
    book.place_bid(price=0.60, quantity=10)
    trades = book.match_orders()
    check("single match: 1 trade",            len(trades) == 1)
    check("single match: price = ask price",  trades[0]['price']    == 0.50)
    check("single match: quantity = 10",      trades[0]['quantity'] == 10)
    check("single match: book empty after",   book.best_bid() is None and book.best_ask() is None)

    # ---------------------------------------------------------------
    # Test 3: Partial fill — bid quantity > ask quantity
    # ---------------------------------------------------------------
    book = OrderBook()
    book.place_ask(price=0.55, quantity=30)
    book.place_bid(price=0.70, quantity=50)
    trades = book.match_orders()
    check("partial fill: 1 trade",              len(trades) == 1)
    check("partial fill: quantity = 30",        trades[0]['quantity'] == 30)
    check("partial fill: bid remainder = 20",   book.best_bid() == 0.70)
    # Remaining bid quantity should be 20
    remaining_qty = -book._bids[0][0]   # peek quantity via price slot trick won't work; check properly
    remaining_qty = book._bids[0][1]
    check("partial fill: remaining bid qty=20", remaining_qty == 20)
    check("partial fill: ask side empty",       book.best_ask() is None)

    # ---------------------------------------------------------------
    # Test 4: Multiple matches in one call
    # ---------------------------------------------------------------
    book = OrderBook()
    book.place_ask(price=0.55, quantity=100)
    book.place_ask(price=0.60, quantity=50)
    book.place_bid(price=0.70, quantity=80)
    book.place_bid(price=0.65, quantity=60)
    trades = book.match_orders()
    total_traded = sum(t['quantity'] for t in trades)
    check("multiple matches: trades > 0",          len(trades) > 0)
    check("multiple matches: total qty = 140",     total_traded == 140)
    check("multiple matches: all prices <= 0.60",  all(t['price'] <= 0.60 for t in trades))

    # ---------------------------------------------------------------
    # Test 5: No match scenario (spread is positive)
    # ---------------------------------------------------------------
    book = OrderBook()
    book.place_bid(price=0.40, quantity=10)
    book.place_ask(price=0.60, quantity=10)
    trades = book.match_orders()
    check("no match: 0 trades",           len(trades) == 0)
    check("no match: bids still present", book.best_bid() == 0.40)
    check("no match: asks still present", book.best_ask() == 0.60)

    # ---------------------------------------------------------------
    # Test 6: Bid exactly equal to ask (should match)
    # ---------------------------------------------------------------
    book = OrderBook()
    book.place_bid(price=0.50, quantity=5)
    book.place_ask(price=0.50, quantity=5)
    trades = book.match_orders()
    check("equal price match: 1 trade",   len(trades) == 1)
    check("equal price match: qty = 5",   trades[0]['quantity'] == 5)

    # ---------------------------------------------------------------
    # Test 7: Best bid / best ask ordering with many orders
    # ---------------------------------------------------------------
    book = OrderBook()
    for p in [0.30, 0.45, 0.50, 0.42]:
        book.place_bid(price=p, quantity=1)
    for p in [0.55, 0.70, 0.60]:
        book.place_ask(price=p, quantity=1)
    check("best bid is highest (0.50)",   book.best_bid() == 0.50)
    check("best ask is lowest  (0.55)",   book.best_ask() == 0.55)
    check("spread = 0.05",                abs(book.spread() - 0.05) < 1e-9)

    # ---------------------------------------------------------------
    # Test 8: Assignment example
    # ---------------------------------------------------------------
    book = OrderBook()
    book.place_ask(price=0.55, quantity=100)
    book.place_ask(price=0.60, quantity=50)
    book.place_bid(price=0.70, quantity=80)
    trades = book.match_orders()
    check("assignment example: 1 trade",         len(trades) == 1)
    check("assignment example: price = 0.55",    trades[0]['price']    == 0.55)
    check("assignment example: quantity = 80",   trades[0]['quantity'] == 80)
    check("assignment example: ask@0.55 qty=20", book.best_ask() == 0.55)
    ask_rem = book._asks[0][1]
    check("assignment example: ask rem qty=20",  ask_rem == 20)
    check("assignment example: no bids left",    book.best_bid() is None)

    print("-" * 55)
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests.")
    print("=" * 55)
    return failed == 0


# ---------------------------------------------------------------------------
# Bonus: Benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    try:
        import matplotlib
        matplotlib.use('TkAgg')   # PyCharm uchun eng ishonchli backend
        import matplotlib.pyplot as plt
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False
        print("matplotlib not found — skipping plot (install with: pip install matplotlib)")
    except Exception:
        try:
            import matplotlib
            matplotlib.use('Agg')  # grafik oynasi chiqmasa ham PNG saqlanadi
            import matplotlib.pyplot as plt
            has_matplotlib = True
        except Exception:
            has_matplotlib = False

    sizes = [100, 1_000, 10_000, 100_000]
    place_times   = []
    match_times   = []

    print("\n" + "=" * 55)
    print("Benchmark")
    print("=" * 55)
    print(f"{'N':>10}  {'Place (s)':>12}  {'Match (s)':>12}")
    print("-" * 40)

    for n in sizes:
        # --- place n/2 bids and n/2 asks (no overlap so nothing matches yet)
        book = OrderBook()
        t0 = time.perf_counter()
        for _ in range(n // 2):
            book.place_bid(round(random.uniform(0.10, 0.45), 4), random.randint(1, 100))
        for _ in range(n // 2):
            book.place_ask(round(random.uniform(0.55, 0.90), 4), random.randint(1, 100))
        t_place = time.perf_counter() - t0
        place_times.append(t_place)

        # --- now add orders that WILL match (bid > ask) and measure match time
        book2 = OrderBook()
        for _ in range(n // 2):
            book2.place_bid(round(random.uniform(0.50, 0.90), 4), random.randint(1, 100))
        for _ in range(n // 2):
            book2.place_ask(round(random.uniform(0.10, 0.50), 4), random.randint(1, 100))
        t0 = time.perf_counter()
        book2.match_orders()
        t_match = time.perf_counter() - t0
        match_times.append(t_match)

        print(f"{n:>10,}  {t_place:>12.6f}  {t_match:>12.6f}")

    print("=" * 55)

    if has_matplotlib:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("OrderBook Benchmark — Heap-based Implementation", fontsize=14)

        axes[0].plot(sizes, place_times, marker='o', color='steelblue', label='measured')
        axes[0].set_title("Time to Place N Orders")
        axes[0].set_xlabel("N (orders)")
        axes[0].set_ylabel("Time (seconds)")
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(True, which='both', linestyle='--', alpha=0.5)

        axes[1].plot(sizes, match_times, marker='s', color='darkorange', label='measured')
        axes[1].set_title("Time to Match N Orders")
        axes[1].set_xlabel("N (orders)")
        axes[1].set_ylabel("Time (seconds)")
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True, which='both', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig("orderbook_benchmark.png", dpi=150)
        print("Grafik saqlandi → orderbook_benchmark.png")
        plt.show(block=True)   # PyCharm da oyna ochiq turadi, yopguncha kutadi


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    all_passed = run_tests()
    run_benchmark()