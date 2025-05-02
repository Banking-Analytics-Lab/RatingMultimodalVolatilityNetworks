import pandas as pd
import numpy as np
import pickle
import networkx as nx
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
import dcor

def load_obj_pickle(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj_pickle(obj, name):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


"""
    Mid-caps can be more sensitive to negative news (asymmetric)
    They might experience more price jumps due to lower liquidity (jump-adjusted)
    Volume can be particularly informative due to varying liquidity (volume-weighted)
"""

def adaptive_volatility(returns, min_window=14, base_window=35, max_window=63):
    """Calculate volatility with adaptive window based on market conditions"""

    # Calculate volatilities for different windows
    short_vol = returns.rolling(min_window).std()
    long_vol = returns.rolling(max_window).std()

    # Calculate volatility ratio
    vol_ratio = short_vol / long_vol

    # Calculate volatilities for all window sizes
    min_window_vol = returns.rolling(min_window).std()
    base_window_vol = returns.rolling(base_window).std()
    max_window_vol = returns.rolling(max_window).std()

    # Create result series
    result = pd.Series(index=returns.index, dtype=float)

    # Fill in values based on conditions
    result[vol_ratio > 1.2] = min_window_vol[vol_ratio > 1.2]
    result[vol_ratio < 0.8] = max_window_vol[vol_ratio < 0.8]
    result[(vol_ratio >= 0.8) & (vol_ratio <= 1.2)] = base_window_vol[(vol_ratio >= 0.8) & (vol_ratio <= 1.2)]

    return result

def asymmetric_volatility(returns, window=30):
    """
        Calculate separate upside and downside volatility
        Captures the fact that markets often "take the stairs up but the elevator down"
        Why? Because markets often react more strongly to negative news than positive news (asymmetric volatility effect).

        In asymmetric_volatility, the 70-30 weighting (0.7 for downside, 0.3 for upside) is based on empirical research that shows market volatility tends to be higher during downturns than during upturns - known as the "leverage effect" or "volatility asymmetry" in finance.
    """

    # Downside volatility
    downside = returns.copy()
    downside[returns > 0] = 0   # Keep only negative returns, set positive to 0
    down_vol = downside.rolling(window).std()   # Calculate std of negative returns

    # Upside volatility
    upside = returns.copy()
    upside[returns < 0] = 0     # Keep only positive returns, set negative to 0
    up_vol = upside.rolling(window).std()   # Calculate std of positive returns

    # Final weighted combination (70% downside, 30% upside)
    return 0.7 * down_vol + 0.3 * up_vol

def jump_adjusted_volatility(returns, window=30, threshold=3):
    """
        Calculate volatility excluding extreme moves
        Reduces the impact of outliers while preserving trend information
        Why? Because extreme jumps can distort volatility estimates. This function "caps" extreme movements while preserving their direction.
    """

    rolling_std = returns.rolling(window).std()
    rolling_mean = returns.rolling(window).mean()

    # Identify extreme moves (jumps)
    # A return is considered a jump if it deviates from the mean by more than 3 standard deviations
    jumps = abs(returns - rolling_mean) > (threshold * rolling_std)

    # Replace extreme moves with capped values:
    # mean + sign(jump) * threshold * std
    adjusted_returns = returns.copy()
    adjusted_returns[jumps] = rolling_mean + np.sign(returns) * threshold * rolling_std

    return adjusted_returns.rolling(window).std()

def volume_weighted_volatility(returns, volume, window=30):
    """
        Calculate volatility giving more weight to high volume days
        Why? Because price movements on high-volume days are often considered more informative than those on low-volume days.
    """

    # Normalize volume relative to its moving average
    vol_weight = volume / volume.rolling(window).mean()

    # A weight > 1 means above-average volume
    # A weight < 1 means below-average volume

    # Weight returns by square root of volume weight
    weighted_returns = returns * np.sqrt(vol_weight)

    # Higher volume days get more weight in volatility calculation

    return weighted_returns.rolling(window).std()

def combined_volatility_measure(returns, volume=None):
    """Combine multiple volatility measures"""

    # Base adaptive volatility
    adapt_vol = adaptive_volatility(returns)

    # Asymmetric component
    asym_vol = asymmetric_volatility(returns)

    # Jump adjustment
    jump_vol = jump_adjusted_volatility(returns)

    # Combine measures
    if volume is not None:
        vol_vol = volume_weighted_volatility(returns, volume)
        return (0.4 * adapt_vol + 0.3 * asym_vol +
                0.2 * jump_vol + 0.1 * vol_vol)
    else:
        return 0.5 * adapt_vol + 0.3 * asym_vol + 0.2 * jump_vol


def distance_correlation(x, y):
    """
    Calculate distance correlation between two series
    Based on Székely et al. (2007)
    """

    dist = dcor.distance_correlation(x, y)

    # Check if dist is nan
    if np.isnan(dist):
        return 0

    return dist

def calculate_distance_correlation(x, y):
    x = np.asarray(x.dropna())
    y = np.asarray(y.dropna())
    common_len = min(len(x), len(y))

    return distance_correlation(x[:common_len], y[:common_len])

def calculate_tail_dependence(x, y, quantile=0.05):
    """
        Calculate tail dependence coefficient using empirical copula
        This function measures how strongly two series are correlated in their extreme values (tails of the distribution).

        This is useful because:
          -  Normal correlation might miss strong relationships in extreme market conditions
          -  High tail dependence means stocks tend to crash or boom together
          -  Important for risk management
    """

    # Convert to arrays and remove NaNs
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = ~(np.isnan(x) | np.isnan(y))

    x = x[mask]
    y = y[mask]

    n = len(x)

    if n < 2:
        return 0.0  # Not enough data

    # Calculate ranks using vectorized operations (faster than pandas)
    rank_x = (np.argsort(np.argsort(x)) + 1) / n
    rank_y = (np.argsort(np.argsort(y)) + 1) / n

    # Precompute thresholds
    upper_threshold = 1 - quantile

    # Vectorized comparisons
    lower_mask = (rank_x <= quantile) & (rank_y <= quantile)
    upper_mask = (rank_x >= upper_threshold) & (rank_y >= upper_threshold)

    # Calculate probabilities
    lower_tail = np.mean(lower_mask) / quantile
    upper_tail = np.mean(upper_mask) / quantile

    return (lower_tail + upper_tail) / 2

def calculate_regime_correlation(volatility_series1, volatility_series2):
    """
    Calculate regime-dependent correlation using volatility states
    Optimized version with mathematical simplifications and vectorization
    """

    x = np.asarray(volatility_series1, dtype=np.float64)
    y = np.asarray(volatility_series2, dtype=np.float64)

    # Remove NaNs using vectorized operations
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    n = len(x)

    if n < 4:  # Need at least 2 points in each regime
        return np.nan

    # Precompute percentiles using partial sorting (faster for large arrays)
    p75_x = np.partition(x, int(n * 0.75))[int(n * 0.75)]
    p75_y = np.partition(y, int(n * 0.75))[int(n * 0.75)]

    # Vectorized mask creation using bitwise OR
    high_vol_mask = (x > p75_x) | (y > p75_y)
    n_high = np.count_nonzero(high_vol_mask)
    n_normal = n - n_high

    # Handle edge cases where we can't compute correlation
    if n_high < 2 or n_normal < 2:
        return np.nan

    # Precompute all necessary sums using matrix algebra
    x_masked = x * high_vol_mask
    y_masked = y * high_vol_mask
    sum_x_high = x_masked.sum()
    sum_y_high = y_masked.sum()
    sum_xy_high = (x * y * high_vol_mask).sum()
    sum_x2_high = (x**2 * high_vol_mask).sum()
    sum_y2_high = (y**2 * high_vol_mask).sum()

    # Compute normal regime sums using complementary mask (no recalc)
    sum_x_normal = x.sum() - sum_x_high
    sum_y_normal = y.sum() - sum_y_high
    sum_xy_normal = (x * y).sum() - sum_xy_high
    sum_x2_normal = (x**2).sum() - sum_x2_high
    sum_y2_normal = (y**2).sum() - sum_y2_high

    # Fast correlation calculation using precomputed sums
    def _corr(sx, sy, sxy, sx2, sy2, n):
        cov = (sxy - (sx * sy) / n) / (n - 1)
        std_x = np.sqrt((sx2 - sx**2 / n) / (n - 1))
        std_y = np.sqrt((sy2 - sy**2 / n) / (n - 1))
        return cov / (std_x * std_y) if (std_x * std_y) > 1e-8 else 0.0

    high_corr = _corr(sum_x_high, sum_y_high, sum_xy_high,
                      sum_x2_high, sum_y2_high, n_high)

    normal_corr = _corr(sum_x_normal, sum_y_normal, sum_xy_normal,
                        sum_x2_normal, sum_y2_normal, n_normal)

    return 0.7 * high_corr + 0.3 * normal_corr

def calculate_edge_weights(volatility_series1, volatility_series2):
    """Calculate enhanced edge weights using multiple measures"""

    # Regular distance correlation
    dcorr = calculate_distance_correlation(volatility_series1, volatility_series2)

    # Tail dependence
    tail_dep = calculate_tail_dependence(volatility_series1, volatility_series2)

    # Regime-based correlation
    regime_corr = calculate_regime_correlation(volatility_series1, volatility_series2)

    # Combine measures
    return 0.5 * dcorr + 0.3 * tail_dep + 0.2 * regime_corr

def handle_price_duplicates(stock_data):
    """
        Handle duplicate prices in stock data using mean values
        Some dates have 2-4 prices for the same ticker
    """

    # Group by date and ticker, calculate mean price and sum volume
    cleaned_data = (stock_data.groupby(['date', 'TICKER']).agg({'PRC': 'mean', 'VOL': 'sum'}).reset_index())

    return cleaned_data

def process_chunk(chunk, volatilities, threshold):
    """Process a chunk of ticker combinations"""

    edges = []

    for i, (t1, t2) in enumerate(chunk):
        print(f"Processing chunk {i+1}/{len(chunk)}")
        weight = calculate_edge_weights(volatilities[t1], volatilities[t2])

        if weight > threshold:
            edges.append((t1, t2, weight))

    return edges

def chunk_combinations(iterable, chunk_size):
    """Split combinations into chunks"""
    chunks = []
    current_chunk = []

    for combo in combinations(iterable, 2):
        current_chunk.append(combo)

        if len(current_chunk) >= chunk_size:
            chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def create_quarterly_graphs_parallel(stock_data, quarterly_data, threshold=0.0, n_workers=None, chunk_size=10000):
    """
    Create graphs for each quarter with parallel processing

    Parameters:
    -----------
    stock_data : pandas DataFrame
    quarterly_data : pandas DataFrame
    threshold : float
    n_workers : int, optional (default: number of CPU cores - 1)
    chunk_size : int, number of combinations to process in each chunk
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    # Clean duplicates
    stock_data_clean = handle_price_duplicates(stock_data)

    # Pre-calculate all returns and volumes in wide format
    returns_data = (stock_data_clean.pivot(index='date',
                                           columns='TICKER',
                                           values='PRC')
                    .pct_change())

    volume_data = stock_data_clean.pivot(index='date',
                                         columns='TICKER',
                                         values='VOL')

    print("Pre-calculating volatilities for all tickers...")
    # Pre-calculate all volatilities for all tickers
    all_volatilities = {}
    for ticker in returns_data.columns:
        all_volatilities[ticker] = combined_volatility_measure(
            returns_data[ticker],
            volume=volume_data[ticker]
        )

    # Get all unique year-quarter combinations
    quarters = quarterly_data.groupby(['year', 'quarter']).size().index
    quarterly_graphs = {}

    for year, quarter in quarters:
        print(f"Creating graph for {year} Q{quarter}")

        # Create graph for this quarter
        G = nx.Graph()

        # Get tickers for this quarter
        quarter_tickers = quarterly_data[
            (quarterly_data['year'] == year) &
            (quarterly_data['quarter'] == quarter)
            ]['tic'].unique()

        # Add nodes first
        G.add_nodes_from(quarter_tickers)

        print("Getting volatilities for this quarter's tickers...")
        # Get volatilities for this quarter's tickers
        quarter_volatilities = {t: all_volatilities[t] for t in quarter_tickers}

        print("Creating chunk combinations...")
        # Split combinations into chunks
        chunks = chunk_combinations(quarter_tickers, chunk_size)

        print(f"Processing {len(chunks)} chunks in parallel...")
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            process_chunk_partial = partial(
                process_chunk,
                volatilities=quarter_volatilities,
                threshold=threshold
            )
            chunk_results = list(executor.map(process_chunk_partial, chunks))

        print("Adding edges to graph...")
        # Add all edges to graph
        for chunk_edges in chunk_results:
            G.add_weighted_edges_from(chunk_edges)

        # Add quarterly statement data as node attributes
        quarter_data = quarterly_data[
            (quarterly_data['year'] == year) &
            (quarterly_data['quarter'] == quarter)
            ]

        print("Adding quarterly statement data as node attributes...")
        node_attrs = {}
        for _, row in quarter_data.iterrows():
            ticker = row['tic']
            node_attrs[ticker] = {
                col: row[col] for col in quarterly_data.columns
                if col not in ['year', 'quarter', 'tic']
            }

        nx.set_node_attributes(G, node_attrs)

        # Store graph
        quarterly_graphs[(year, quarter)] = G

        print(f"Created graph for {year} Q{quarter} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        print("Saving graph to file...")
        # Save graph to file
        save_obj_pickle(G, f'../Data/quarterly_graphs/{year}_Q{quarter}_graph')
        print()

    return quarterly_graphs

def main():
    # Load the data
    stock_data = load_obj_pickle('../Data/stock_data_processed')
    quarterly_data = load_obj_pickle('../Data/quarterly_data_standardized')

    # Sort quarterly data by datadate and reset index
    quarterly_data = quarterly_data.sort_values(by=['datadate']).reset_index(drop=True)

    # Drop datadate column
    quarterly_data = quarterly_data.drop(columns=['datadate'])

    # Create graphs using parallel processing
    quarterly_graphs = create_quarterly_graphs_parallel(
        stock_data=stock_data,
        quarterly_data=quarterly_data,
        n_workers=None,
        chunk_size=10000  # Process 10,000 combinations at a time
    )

    # Save the quarterly graphs
    save_obj_pickle(quarterly_graphs, '../Data/quarterly_graphs')

if __name__ == '__main__':
    main()