"""
Core analysis functions - shared between MCP server and Gradio app.
"""

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Global dataframes storage
_dataframes: dict[str, pd.DataFrame] = {}


def setup_plot_style():
    """Set up dark theme for plots."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#1a1a1a',
        'axes.facecolor': '#1a1a1a',
        'axes.edgecolor': '#444',
        'axes.labelcolor': '#ccc',
        'text.color': '#ccc',
        'xtick.color': '#999',
        'ytick.color': '#999',
        'grid.color': '#333',
        'legend.facecolor': '#1a1a1a',
        'legend.edgecolor': '#444',
    })


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
               facecolor='#1a1a1a', edgecolor='none')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def load_datasets():
    """Load datasets from data directory."""
    if not DATA_DIR.exists():
        return

    datasets = {
        'air_quality': 'sample_air_quality.csv',
        'funding': 'funding_data.csv',
        'city_info': 'city_info.csv',
    }

    for name, filename in datasets.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            try:
                _dataframes[name] = pd.read_csv(filepath)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")


def get_dataframes():
    """Get the loaded dataframes dict."""
    if not _dataframes:
        load_datasets()
    return _dataframes


# =============================================================================
# DATA EXPLORATION
# =============================================================================

def list_tables() -> str:
    """List all available tables."""
    dfs = get_dataframes()
    if not dfs:
        return "No tables loaded."

    descriptions = {
        'air_quality': 'Hourly air quality readings (PM2.5, PM10, NO2, SO2, CO, O3) for Delhi & Bangalore',
        'funding': 'Government funding for air quality initiatives by city and year',
        'city_info': 'City metadata - population, vehicles, industries, green cover',
    }

    lines = ["# Available Tables", ""]
    for name, df in dfs.items():
        desc = descriptions.get(name, "No description")
        lines.append(f"## {name}")
        lines.append(f"- **Description:** {desc}")
        lines.append(f"- **Rows:** {len(df):,}")
        lines.append(f"- **Columns:** {', '.join(df.columns[:8])}")
        if len(df.columns) > 8:
            lines.append(f"  ... and {len(df.columns) - 8} more")
        lines.append("")

    return "\n".join(lines)


def show_table(name: str, rows: int = 10, columns: list[str] | None = None) -> str:
    """Display rows from a table."""
    dfs = get_dataframes()
    if name not in dfs:
        available = ", ".join(dfs.keys())
        return f"Table '{name}' not found. Available: {available}"

    df = dfs[name]
    if columns:
        df = df[columns]

    return f"## {name} (showing {min(rows, len(df))} of {len(df)} rows)\n\n{df.head(rows).to_markdown(index=False)}"


def describe_table(name: str) -> str:
    """Get detailed statistics for a table."""
    dfs = get_dataframes()
    if name not in dfs:
        available = ", ".join(dfs.keys())
        return f"Table '{name}' not found. Available: {available}"

    df = dfs[name]

    lines = [f"# {name} - Detailed Info", ""]
    lines.append(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    lines.append("")
    lines.append("## Columns")

    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        unique = df[col].nunique()

        lines.append(f"\n### {col}")
        lines.append(f"- Type: {dtype}")
        lines.append(f"- Non-null: {non_null:,} ({non_null/len(df)*100:.1f}%)")
        lines.append(f"- Unique: {unique:,}")

        if pd.api.types.is_numeric_dtype(df[col]):
            lines.append(f"- Range: {df[col].min():.2f} to {df[col].max():.2f}")
            lines.append(f"- Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")
        elif df[col].dtype == 'object' and unique <= 10:
            lines.append(f"- Values: {', '.join(map(str, df[col].unique()))}")

    return "\n".join(lines)


def query_table(name: str, query: str) -> str:
    """Filter a table using pandas query syntax."""
    dfs = get_dataframes()
    if name not in dfs:
        available = ", ".join(dfs.keys())
        return f"Table '{name}' not found. Available: {available}"

    df = dfs[name]

    try:
        result = df.query(query)
        n_results = len(result)

        if n_results == 0:
            return f"No rows match query: {query}"

        lines = [f"## Query Results ({n_results:,} rows)", f"Query: `{query}`", ""]

        if n_results <= 20:
            lines.append(result.to_markdown(index=False))
        else:
            lines.append(result.head(20).to_markdown(index=False))
            lines.append(f"\n*...and {n_results - 20:,} more rows*")

        return "\n".join(lines)

    except Exception as e:
        return f"Query error: {e}"


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compare_weekday_weekend(value_column: str, group_by: str | None = None,
                            table: str = "air_quality") -> str:
    """Compare weekday vs weekend values."""
    dfs = get_dataframes()
    if table not in dfs:
        return f"Table '{table}' not found."

    df = dfs[table].copy()

    if 'day_of_week' not in df.columns:
        return "Table doesn't have 'day_of_week' column"

    weekend_days = ['Saturday', 'Sunday']
    df['_day_type'] = df['day_of_week'].apply(lambda x: 'Weekend' if x in weekend_days else 'Weekday')

    lines = [f"# Weekday vs Weekend: {value_column}", ""]

    if group_by and group_by in df.columns:
        pivot = df.pivot_table(values=value_column, index=group_by,
                               columns='_day_type', aggfunc='mean').round(2)
        pivot['Change'] = pivot['Weekend'] - pivot['Weekday']
        pivot['Change %'] = ((pivot['Weekend'] - pivot['Weekday']) / pivot['Weekday'] * 100).round(1)

        lines.append(pivot.to_markdown())
    else:
        stats = df.groupby('_day_type')[value_column].agg(['mean', 'std', 'count']).round(2)
        lines.append(stats.to_markdown())

        diff = stats.loc['Weekend', 'mean'] - stats.loc['Weekday', 'mean']
        pct = diff / stats.loc['Weekday', 'mean'] * 100
        lines.append(f"\n**Change:** {diff:.2f} ({pct:.1f}%)")

    return "\n".join(lines)


def compare_cities(value_column: str, cities: list[str] | None = None,
                   table: str = "air_quality") -> str:
    """Compare a metric across cities."""
    dfs = get_dataframes()
    if table not in dfs:
        return f"Table '{table}' not found."

    df = dfs[table]

    if 'city' not in df.columns:
        return "Table doesn't have 'city' column"

    if cities:
        df = df[df['city'].isin(cities)]

    stats = df.groupby('city')[value_column].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(2).sort_values('mean', ascending=False)

    lines = [f"# {value_column} by City", "", stats.to_markdown()]

    return "\n".join(lines)


def analyze_correlation(columns: list[str] | None = None, target: str | None = None,
                        table: str = "air_quality") -> str:
    """Analyze correlations between numeric columns."""
    dfs = get_dataframes()
    if table not in dfs:
        return f"Table '{table}' not found."

    df = dfs[table]

    if columns:
        numeric_df = df[columns].select_dtypes(include=[np.number])
    else:
        numeric_df = df.select_dtypes(include=[np.number])

    corr = numeric_df.corr().round(3)

    if target and target in corr.columns:
        target_corr = corr[target].drop(target).sort_values(key=abs, ascending=False)
        lines = [f"# Correlations with {target}", ""]
        for col, val in target_corr.items():
            strength = "strong" if abs(val) > 0.7 else "moderate" if abs(val) > 0.4 else "weak"
            lines.append(f"- **{col}:** {val:.3f} ({strength})")
        return "\n".join(lines)
    else:
        return f"# Correlation Matrix\n\n{corr.to_markdown()}"


def analyze_funding(city: str | None = None, year: int | None = None) -> str:
    """Analyze air quality funding data."""
    dfs = get_dataframes()
    if 'funding' not in dfs:
        return "Funding table not loaded."

    df = dfs['funding']

    if city:
        df = df[df['city'] == city]
    if year:
        df = df[df['year'] == year]

    if len(df) == 0:
        return "No data matches the filters."

    lines = ["# Funding Analysis", ""]

    if city and not year:
        lines.append(f"## {city} - Yearly Breakdown")
        lines.append(df.to_markdown(index=False))

        total = df['total_budget_cr'].sum()
        growth = (df.iloc[-1]['total_budget_cr'] / df.iloc[0]['total_budget_cr'] - 1) * 100
        lines.append(f"\n**Total (all years):** ₹{total:.1f} Cr")
        lines.append(f"**Growth:** {growth:.1f}%")

    elif year and not city:
        lines.append(f"## Year {year} - All Cities")
        lines.append(df.to_markdown(index=False))

        total = df['total_budget_cr'].sum()
        lines.append(f"\n**Total allocation:** ₹{total:.1f} Cr")

    else:
        by_city = df.groupby('city')['total_budget_cr'].sum().sort_values(ascending=False)
        by_year = df.groupby('year')['total_budget_cr'].sum()

        lines.append("## Total Budget by City")
        lines.append(by_city.to_markdown())
        lines.append("\n## Total Budget by Year")
        lines.append(by_year.to_markdown())

    return "\n".join(lines)


def get_city_profile(city: str) -> str:
    """Get comprehensive profile for a city."""
    dfs = get_dataframes()
    lines = [f"# {city} - City Profile", ""]

    # City info
    if 'city_info' in dfs:
        info = dfs['city_info']
        city_row = info[info['city'] == city]
        if len(city_row) > 0:
            row = city_row.iloc[0]
            lines.append("## Demographics & Infrastructure")
            lines.append(f"- **Population:** {row['population_millions']:.1f} million")
            lines.append(f"- **Area:** {row['area_sq_km']:,} sq km")
            lines.append(f"- **Vehicles:** {row['vehicles_lakhs']} lakhs")
            lines.append(f"- **Industries:** {row['industries']:,}")
            lines.append(f"- **Green Cover:** {row['green_cover_pct']}%")
            lines.append(f"- **Days exceeding WHO limit:** {row['who_limit_days_exceeded']}")
            lines.append("")

    # Air quality summary
    if 'air_quality' in dfs:
        aq = dfs['air_quality']
        city_aq = aq[aq['city'] == city]
        if len(city_aq) > 0:
            lines.append("## Air Quality Summary")
            for col in ['PM2.5', 'PM10', 'NO2', 'AQI_category']:
                if col in city_aq.columns:
                    if col == 'AQI_category':
                        top_cat = city_aq[col].value_counts().head(3)
                        lines.append(f"- **AQI Categories:** {', '.join(f'{k}({v})' for k,v in top_cat.items())}")
                    else:
                        lines.append(f"- **{col}:** Mean={city_aq[col].mean():.1f}, Max={city_aq[col].max():.1f}")
            lines.append("")

    # Funding summary
    if 'funding' in dfs:
        fund = dfs['funding']
        city_fund = fund[fund['city'] == city]
        if len(city_fund) > 0:
            lines.append("## Funding Summary")
            total = city_fund['total_budget_cr'].sum()
            latest = city_fund[city_fund['year'] == city_fund['year'].max()]['total_budget_cr'].iloc[0]
            lines.append(f"- **Total Budget (all years):** ₹{total:.1f} Cr")
            lines.append(f"- **Latest Year Budget:** ₹{latest:.1f} Cr")

    return "\n".join(lines)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_comparison(value_column: str, group_column: str = "city",
                    chart_type: str = "bar", table: str = "air_quality",
                    title: str | None = None) -> str:
    """Create a comparison chart."""
    dfs = get_dataframes()
    if table not in dfs:
        return f"Table '{table}' not found."

    df = dfs[table]
    setup_plot_style()
    plt.close('all')

    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == 'box':
        groups = df[group_column].unique()
        data = [df[df[group_column] == g][value_column].dropna() for g in groups]
        bp = ax.boxplot(data, labels=groups, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    else:
        grouped = df.groupby(group_column)[value_column].mean().sort_values(ascending=False)
        colors = plt.cm.Set2(np.linspace(0, 1, len(grouped)))

        if chart_type == 'horizontal_bar':
            ax.barh(grouped.index, grouped.values, color=colors)
            ax.set_xlabel(value_column)
        else:
            bars = ax.bar(grouped.index, grouped.values, color=colors)
            ax.set_ylabel(value_column)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')
            plt.xticks(rotation=45, ha='right')

    ax.set_title(title or f'{value_column} by {group_column}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    img_base64 = fig_to_base64(fig)
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"


def plot_time_series(value_column: str, group_by: str | None = None,
                     table: str = "air_quality", title: str | None = None) -> str:
    """Create a time series plot."""
    dfs = get_dataframes()
    if table not in dfs:
        return f"Table '{table}' not found."

    df = dfs[table].copy()

    if 'date' not in df.columns:
        return "Table doesn't have 'date' column"

    df['_date'] = pd.to_datetime(df['date'])
    setup_plot_style()
    plt.close('all')

    fig, ax = plt.subplots(figsize=(12, 6))

    if group_by and group_by in df.columns:
        groups = df[group_by].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        for group, color in zip(groups, colors):
            group_df = df[df[group_by] == group]
            daily = group_df.groupby('_date')[value_column].mean()
            ax.plot(daily.index, daily.values, label=group, color=color, linewidth=2)
        ax.legend()
    else:
        daily = df.groupby('_date')[value_column].mean()
        ax.plot(daily.index, daily.values, color='#3498db', linewidth=2)

    ax.set_xlabel('Date')
    ax.set_ylabel(value_column)
    ax.set_title(title or f'{value_column} Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    img_base64 = fig_to_base64(fig)
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"


def plot_weekday_weekend(value_column: str, group_by: str | None = "city",
                         table: str = "air_quality", title: str | None = None) -> str:
    """Create weekday vs weekend comparison chart."""
    dfs = get_dataframes()
    if table not in dfs:
        return f"Table '{table}' not found."

    df = dfs[table].copy()
    setup_plot_style()
    plt.close('all')

    weekend_days = ['Saturday', 'Sunday']
    df['_day_type'] = df['day_of_week'].apply(lambda x: 'Weekend' if x in weekend_days else 'Weekday')

    fig, ax = plt.subplots(figsize=(10, 6))

    if group_by and group_by in df.columns:
        groups = df[group_by].unique()
        x = np.arange(len(groups))
        width = 0.35

        weekday_vals = [df[(df[group_by] == g) & (df['_day_type'] == 'Weekday')][value_column].mean() for g in groups]
        weekend_vals = [df[(df[group_by] == g) & (df['_day_type'] == 'Weekend')][value_column].mean() for g in groups]

        bars1 = ax.bar(x - width/2, weekday_vals, width, label='Weekday', color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x + width/2, weekend_vals, width, label='Weekend', color='#3498db', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(groups)

        for bar in bars1:
            ax.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')
        for bar in bars2:
            ax.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')

    ax.set_ylabel(value_column)
    ax.set_title(title or f'Weekday vs Weekend: {value_column}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    img_base64 = fig_to_base64(fig)
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"


def plot_funding_trend(cities: list[str] | None = None, title: str | None = None) -> str:
    """Plot funding trends over years by city."""
    dfs = get_dataframes()
    if 'funding' not in dfs:
        return "Funding table not loaded."

    df = dfs['funding']

    if cities:
        df = df[df['city'].isin(cities)]

    setup_plot_style()
    plt.close('all')

    fig, ax = plt.subplots(figsize=(12, 6))

    for city in df['city'].unique():
        city_df = df[df['city'] == city].sort_values('year')
        ax.plot(city_df['year'], city_df['total_budget_cr'], marker='o',
               linewidth=2, markersize=8, label=city)

    ax.set_xlabel('Year')
    ax.set_ylabel('Total Budget (₹ Cr)')
    ax.set_title(title or 'Air Quality Funding by City', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    img_base64 = fig_to_base64(fig)
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"


def plot_hourly_pattern(value_column: str, group_by: str | None = "city",
                        table: str = "air_quality", title: str | None = None) -> str:
    """Plot hourly patterns."""
    dfs = get_dataframes()
    if table not in dfs:
        return f"Table '{table}' not found."

    df = dfs[table]
    setup_plot_style()
    plt.close('all')

    fig, ax = plt.subplots(figsize=(12, 6))

    if group_by and group_by in df.columns:
        for city in df[group_by].unique():
            city_df = df[df[group_by] == city]
            hourly = city_df.groupby('hour')[value_column].mean()
            ax.plot(hourly.index, hourly.values, marker='o', linewidth=2,
                   markersize=4, label=city)
        ax.legend()
    else:
        hourly = df.groupby('hour')[value_column].mean()
        ax.plot(hourly.index, hourly.values, marker='o', linewidth=2,
               markersize=6, color='#3498db')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel(value_column)
    ax.set_title(title or f'{value_column} by Hour', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    img_base64 = fig_to_base64(fig)
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"


# Load datasets on module import
load_datasets()
