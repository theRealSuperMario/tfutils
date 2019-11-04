## Installation

```bash
git clone https://github.com/theRealSuperMario/tfutils.git
cd tfutils
pip install .
```

## Runing tests

* [pytest-mpl][https://github.com/matplotlib/pytest-mpl] is used. To run tests, first generate baseline images.
```bast
py.test --mpl-generate-path=baseline
```
Then, make changes and run test
```bash
py.test --mpl --mpl-baseline-path=baseline
```