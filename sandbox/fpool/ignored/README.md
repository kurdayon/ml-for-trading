# Scripts and notes excluded from the library

This folder preserves miscellaneous helpers, experiments, and design notes that are not part
of the supported `fpool` library.

| File | Purpose |
| --- | --- |
| [random_string.py](random_string.py) | Generates random ASCII lowercase text for examples or throwaway labels. The function defaults to 20 characters; running the script prints a 30-character example. Results are neither guaranteed unique nor suitable for passwords or security tokens. |
| [nested_cross_validation.md](nested_cross_validation.md) | Design note from `meta_fold.py` explaining how nested cross-validation separates model selection from evaluation, with considerations for time-series forecasting. No implementation is included. |

Run the example from the repository root:

```sh
python3 sandbox/fpool/ignored/random_string.py
```
