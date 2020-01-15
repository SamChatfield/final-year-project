import sys

import numpy as np

EXAMPLE_X = 2
EXAMPLE_Y = np.array(['a', 'b', 'c'])
EXAMPLE_A = np.array([
    [0.5, 0.5],
    [0.3, 0.7],
])
EXAMPLE_B = np.array([
    [0.6, 0.4, 0.0],
    [0.0, 0.8, 0.2],
])
EXAMPLE_S = np.array([1.0, 0.0])

# TODO: Create HMMParams class to wrap [x, y, a, b, s]


class HMM:
    def __init__(self,
                 x=EXAMPLE_X,
                 y=EXAMPLE_Y,
                 a=EXAMPLE_A,
                 b=EXAMPLE_B,
                 s=EXAMPLE_S,
                 debug=False):
        # Parameter assertions
        assert a.shape == (x, x)
        assert b.shape == (x, y.size)
        assert s.shape == (x, )

        # Parameters
        self.init_x = x
        self.init_y = y
        # Normalise the rows of A and B to sum to 1
        self.init_a = a
        # self.init_a = a / a.sum(axis=1, keepdims=True)
        self.init_b = b
        # self.init_b = b / b.sum(axis=1, keepdims=True)
        self.init_s = s

        # Options
        self._debug = debug

        self.reset()

    def reset(self):
        self.x = self.init_x
        self.y = self.init_y
        self.a = self.init_a
        self.b = self.init_b
        self.s = self.init_s

        self._current_state = np.random.choice(self.x, p=self.s)
        if self._debug:
            print(f'Start:\t{self._current_state}\n')

    def simulate(self, n=1, reset_before=False):
        if reset_before:
            self.reset()

        states = []
        symbols = []

        for _ in range(n):
            if self._debug:
                print(f'Curr:\t{self._current_state}')

            # Add current state to history of states
            states.append(self._current_state)

            # Emission
            p_em = self.b[self._current_state]
            o = np.random.choice(self.y, p=p_em)
            if self._debug:
                print(f'Emit:\t{o}')
            symbols.append(o)

            # Transition
            p_trans = self.a[self._current_state]
            ns = np.random.choice(self.x, p=p_trans)
            if self._debug:
                print(f'Next:\t{ns}\n')
            self._current_state = ns

        return (np.array(states), np.array(symbols))

    def _fwd_alpha(t, i):
        pass

    def _symbol_idx(self, symbol):
        where_res = np.where(self.y == symbol)
        assert len(where_res) == 1

        idxs = where_res[0]
        assert idxs.shape == (1,)
        assert idxs.size == 1

        idx = idxs[0]
        return idx

    def fwd(self, state_seq, symbol_seq):
        """Forward algorithm on this HMM (uses transition and emission matrices)"""
        # print(f'Fwd for st: {state_seq}, sy: {symbol_seq}')

        for t in range(0, symbol_seq.size):
            sym = symbol_seq[t]
            sym_idx = self._symbol_idx(sym)
            # print(f't={t}, sym={sym}, sym_idx={sym_idx}')
            # Initial alpha
            if t == 0:
                alpha = np.multiply(self.s, self.b[:, sym_idx])
            else:
                alpha = np.dot(alpha, self.a)
                alpha = np.multiply(alpha, self.b[:, sym_idx])
            # print(f't={t}, alpha={alpha}')

        # print(f'Alpha at end: {alpha}')
        return alpha.sum()


def main(steps):
    h = HMM()
    l = h.simulate(steps)
    print(''.join(l))


if __name__ == '__main__':
    main(int(sys.argv[1]))
