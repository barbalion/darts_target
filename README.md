# Dartboard Heatmap Visualization

This project generates heatmaps of a dartboard to visualize expected scores based on a player's aim precision (`sigma`). The heatmaps highlight optimal aiming points, with the top 5% of expected scores colored in a green gradient and the lower 95% colored from black to white.

## Features

- Simulates dart throws with varying levels of aim precision.
- Calculates expected scores by averaging multiple random samples per point.
- Generates heatmaps highlighting top scoring areas on the dartboard.
- Outputs progress in the console with elapsed and estimated times.
- Saves generated images as `boardXXX.png`, where `XXX` is the zero-padded `sigma` value.

## Installation

### Prerequisites

- Python 3.6 or higher
- `matplotlib` library
- `numpy` library

### Install Dependencies

```bash
pip install matplotlib numpy
```

## Usage

1. **Run the Script**

   ```bash
   python darts.py
   ```

   The script will generate heatmaps for `sigma` values from 1 to 100 without displaying the graphs.

2. **Monitor Progress**

   The console will display progress information, including the current `sigma` value, elapsed time, and estimated remaining time.

## Output

- **Console Output**: Displays progress updates after each `sigma` value is processed.
- **Generated Images**: Saves images as `boardXXX.png` for each `sigma` value, highlighting optimal aiming points on the dartboard.

## License

This project is licensed under the [MIT License](LICENSE).