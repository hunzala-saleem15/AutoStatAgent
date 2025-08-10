import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

class visualization:
    @staticmethod
    def _top_categories(series, top_k=8):
        top = series.value_counts(dropna=False).nlargest(top_k).index
        return series.where(series.isin(top), other="Other")

    @staticmethod
    def create_visualizations(
        df,
        output_dir='output/plots',
        verbose=False,
        max_scatter_pairs=6,
        max_boxplots=20,
        max_cat_countplots=12
    ):
        os.makedirs(output_dir, exist_ok=True)
        plot_paths = {}

        n_rows, n_cols = df.shape
        if verbose:
            print(f"Dataset shape: {n_rows} rows x {n_cols} cols")

        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if (pd.api.types.is_categorical_dtype(df[c]) or df[c].dtype == object)]
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

        def is_informative(col):
            nunique = df[col].nunique(dropna=False)
            if nunique <= 1:
                return False
            if col in cat_cols and nunique > 50:
                return False
            if col in num_cols and nunique == n_rows:
                return False
            return True

        num_cols = [c for c in num_cols if is_informative(c)]
        cat_cols = [c for c in cat_cols if is_informative(c)]
        dt_cols = [c for c in dt_cols if is_informative(c)]

        if verbose:
            print(f"Numeric cols: {num_cols}")
            print(f"Categorical cols: {cat_cols}")
            print(f"Datetime cols: {dt_cols}")

        # Numeric univariate
        for col in num_cols:
            try:
                series = df[col].dropna()
                if series.empty or series.std() == 0:
                    continue

                plt.figure()
                sns.histplot(series, kde=True, stat="density", edgecolor=None)
                plt.title(f"{col} Distribution")
                hist_path = os.path.join(output_dir, f"{col}_hist.png")
                plt.savefig(hist_path, bbox_inches='tight')
                plt.close()
                plot_paths[f'{col}_histogram'] = hist_path

                plt.figure(figsize=(6, 3))
                sns.boxplot(x=series)
                plt.title(f"{col} Boxplot")
                box_path = os.path.join(output_dir, f"{col}_box.png")
                plt.savefig(box_path, bbox_inches='tight')
                plt.close()
                plot_paths[f'{col}_boxplot'] = box_path

                if verbose:
                    print(f"Created numeric plots for {col}")
            except Exception as e:
                if verbose:
                    print(f"Error in numeric univariate for {col}: {e}")

        # Categorical univariate
        for col in cat_cols:
            try:
                if df[col].nunique(dropna=False) > max_cat_countplots:
                    ser = visualization._top_categories(df[col], top_k=10)
                    title_extra = " (top 10 + Other)"
                else:
                    ser = df[col].fillna("NaN")
                    title_extra = ""

                plt.figure(figsize=(6, 4))
                order = ser.value_counts().index
                sns.countplot(x=ser, order=order)
                plt.xticks(rotation=45, ha='right')
                plt.title(f"{col} Counts{title_extra}")
                count_path = os.path.join(output_dir, f"{col}_count.png")
                plt.tight_layout()
                plt.savefig(count_path, bbox_inches='tight')
                plt.close()
                plot_paths[f'{col}_countplot'] = count_path

                if verbose:
                    print(f"Created categorical countplot for {col}")
            except Exception as e:
                if verbose:
                    print(f"Error in categorical univariate for {col}: {e}")

        # Correlation heatmap and scatterplots for top correlated pairs
        if len(num_cols) > 1:
            try:
                corr = df[num_cols].corr()
                plt.figure(figsize=(min(12, len(num_cols)), min(10, len(num_cols))))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
                plt.title("Numeric Correlation Matrix")
                heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
                plt.savefig(heatmap_path, bbox_inches='tight')
                plt.close()
                plot_paths['correlation_heatmap'] = heatmap_path
                if verbose:
                    print("Created correlation heatmap")

                corr_abs = corr.abs().where(~np.eye(len(corr), dtype=bool))
                pairs = (corr_abs.stack()
                         .reset_index()
                         .rename(columns={'level_0': 'x', 'level_1': 'y', 0: 'abs_corr'}))
                pairs = pairs[pairs['x'] < pairs['y']]
                pairs = pairs.sort_values('abs_corr', ascending=False)

                scatter_count = 0
                for _, row in pairs.iterrows():
                    if scatter_count >= max_scatter_pairs:
                        break
                    if row['abs_corr'] < 0.3:
                        break
                    x, y = row['x'], row['y']
                    plt.figure(figsize=(6, 4))
                    sns.scatterplot(x=df[x], y=df[y], alpha=0.6)
                    plt.title(f"{x} vs {y} (|r|={row['abs_corr']:.2f})")
                    scatter_path = os.path.join(output_dir, f"{x}_vs_{y}_scatter.png")
                    plt.savefig(scatter_path, bbox_inches='tight')
                    plt.close()
                    plot_paths[f'{x}_vs_{y}_scatter'] = scatter_path
                    scatter_count += 1

                if verbose:
                    print(f"Created {scatter_count} scatterplots")
            except Exception as e:
                if verbose:
                    print(f"Error creating correlation/scatter plots: {e}")

        # Numeric by categorical boxplots (limit and only small-cardinality cats)
        boxplot_count = 0
        for num_col in num_cols:
            for cat_col in cat_cols:
                if boxplot_count >= max_boxplots:
                    break
                if df[cat_col].nunique(dropna=False) <= 10:
                    try:
                        group_sizes = df.groupby(cat_col)[num_col].size()
                        if (group_sizes < 3).sum() > len(group_sizes) * 0.5:
                            continue

                        plt.figure(figsize=(8, 4))
                        sns.boxplot(x=visualization._top_categories(df[cat_col], top_k=10), y=df[num_col])
                        plt.xticks(rotation=45, ha='right')
                        plt.title(f"{num_col} by {cat_col}")
                        box_path = os.path.join(output_dir, f"{num_col}_by_{cat_col}_box.png")
                        plt.tight_layout()
                        plt.savefig(box_path, bbox_inches='tight')
                        plt.close()
                        plot_paths[f'{num_col}_by_{cat_col}_boxplot'] = box_path
                        boxplot_count += 1

                        if verbose:
                            print(f"Created boxplot for {num_col} by {cat_col}")
                    except Exception as e:
                        if verbose:
                            print(f"Error boxplot {num_col} by {cat_col}: {e}")
            if boxplot_count >= max_boxplots:
                break

        # Categorical vs categorical heatmaps (only small cardinalities)
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                a, b = cat_cols[i], cat_cols[j]
                if df[a].nunique(dropna=False) <= 8 and df[b].nunique(dropna=False) <= 8:
                    try:
                        cross_tab = pd.crosstab(df[a], df[b])
                        plt.figure(figsize=(6, 5))
                        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues')
                        plt.title(f"{a} vs {b}")
                        heatmap_path = os.path.join(output_dir, f"{a}_vs_{b}_heatmap.png")
                        plt.tight_layout()
                        plt.savefig(heatmap_path, bbox_inches='tight')
                        plt.close()
                        plot_paths[f'{a}_vs_{b}_heatmap'] = heatmap_path
                        if verbose:
                            print(f"Created heatmap for {a} vs {b}")
                    except Exception as e:
                        if verbose:
                            print(f"Error heatmap for {a} vs {b}: {e}")

        # Datetime trends: monthly counts + monthly mean for top numeric cols
        for dt_col in dt_cols:
            try:
                df_dt = df.dropna(subset=[dt_col])
                if df_dt.empty:
                    continue

                ts = df_dt.set_index(dt_col).resample('M').size()
                if ts.sum() > 0 and len(ts) > 1:
                    plt.figure(figsize=(8, 3))
                    ts.plot(marker='o')
                    plt.title(f"Monthly record count ({dt_col})")
                    plt.ylabel("Count")
                    path = os.path.join(output_dir, f"{dt_col}_monthly_count.png")
                    plt.tight_layout()
                    plt.savefig(path, bbox_inches='tight')
                    plt.close()
                    plot_paths[f'{dt_col}_monthly_count'] = path
                    if verbose:
                        print(f"Created monthly count plot for {dt_col}")

                numeric_sample = [c for c in num_cols if df_dt[c].dropna().shape[0] > 10][:3]
                for num_col in numeric_sample:
                    monthly_mean = df_dt.set_index(dt_col)[num_col].resample('M').mean()
                    if monthly_mean.dropna().empty:
                        continue
                    plt.figure(figsize=(8, 3))
                    monthly_mean.plot(marker='o')
                    plt.title(f"Monthly mean of {num_col} by {dt_col}")
                    plt.ylabel(num_col)
                    path = os.path.join(output_dir, f"{num_col}_monthly_mean_by_{dt_col}.png")
                    plt.tight_layout()
                    plt.savefig(path, bbox_inches='tight')
                    plt.close()
                    plot_paths[f'{num_col}_monthly_mean_by_{dt_col}'] = path
                    if verbose:
                        print(f"Created monthly mean for {num_col} by {dt_col}")

            except Exception as e:
                if verbose:
                    print(f"Error in datetime analysis for {dt_col}: {e}")

        return plot_paths
