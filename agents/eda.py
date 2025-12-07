"""
Module: eda.py
Description: Handles comprehensive Exploratory Data Analysis (EDA) in a modular and agent-friendly way.
Author: Raj Nandini
Date: 2025-10-01
"""
import matplotlib
matplotlib.use("Agg")   # ← prevents GUI errors in background threads

import os
import matplotlib.pyplot as plt
plt.ioff()  
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2_contingency
from orchestrator.metrics_tracker import metrics


from orchestrator.pipeline_context import PipelineContext  

class EDAModule:
    def __init__(self, context: PipelineContext, llm_agent=None, save_plots=True, plot_dir="eda_plots"):
        self.context = context
        self.llm_agent = llm_agent
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        metrics.task_executed()

    #############################
    # 1. Data Overviewz
    #############################
    def data_overview(self, df):
        overview = {
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isna().sum().to_dict(),
            "memory_usage_bytes": df.memory_usage().sum()
        }
        metrics.auto_mod()
        return overview

    #############################
    # 2. Univariate Analysis
    #############################
    def univariate_analysis(self, df):
        results = {}
        metrics.auto_mod()
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(include="object").columns

        for col in num_cols:
            plt.figure()
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            if self.save_plots: plt.savefig(os.path.join(self.plot_dir, f"{col}_hist.png"))
            plt.close()

            plt.figure()
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot of {col}")
            if self.save_plots: plt.savefig(os.path.join(self.plot_dir, f"{col}_box.png"))
            plt.close()

            plt.figure()
            sns.violinplot(x=df[col])
            plt.title(f"Violin Plot of {col}")
            if self.save_plots: plt.savefig(os.path.join(self.plot_dir, f"{col}_violin.png"))
            plt.close()

        for col in cat_cols:
            plt.figure(figsize=(8,4))
            df[col].value_counts().plot(kind="bar")
            plt.title(f"Categorical Distribution: {col}")
            if self.save_plots: plt.savefig(os.path.join(self.plot_dir, f"{col}_bar.png"))
            plt.close()

        return {"numeric_columns": list(num_cols), "categorical_columns": list(cat_cols)}

    #############################
    # 3. Bivariate / Multivariate Analysis
    #############################
    def bivariate_analysis(self, df, target_col=None):
        results = {}
        metrics.auto_mod()
        # Correlation matrix for numeric columns
        corr = df.corr(numeric_only=True)
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, cmap="coolwarm", annot=False)
        plt.title("Correlation Heatmap")
        if self.save_plots: plt.savefig(os.path.join(self.plot_dir, "correlation_heatmap.png"))
        plt.close()
        results["correlation_matrix"] = corr.to_dict()

        # Pairplot (subset for performance)
        if df.select_dtypes(include=np.number).shape[1] <= 6:
            sns.pairplot(df.select_dtypes(include=np.number))
            if self.save_plots: plt.savefig(os.path.join(self.plot_dir, "pairplot.png"))
            plt.close()

        # Chi-square test for categorical vs target
        chi_results = {}
        cat_cols = df.select_dtypes(include="object").columns
        if target_col and target_col in df.columns:
            for col in cat_cols:
                if col != target_col:
                    contingency = pd.crosstab(df[col], df[target_col])
                    chi2, p, dof, ex = chi2_contingency(contingency)
                    chi_results[col] = {"chi2": chi2, "p_value": p}
        results["chi_square_tests"] = chi_results

        return results

    #############################
    # 4. Target Distribution Check
    #############################
    def target_analysis(self, df, target_col):
        if not target_col or target_col not in df.columns:
            return {}
        metrics.auto_mod()
        target_counts = df[target_col].value_counts(normalize=True).to_dict()
        imbalance_flag = min(target_counts.values()) < 0.2
        return {"target_distribution": target_counts, "imbalance_flag": imbalance_flag}

    #############################
    # 5. Outlier & Anomaly Visualization
    #############################
    def outlier_analysis(self, df):
        results = {}
        metrics.auto_mod()
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            plt.figure()
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot: {col}")
            if self.save_plots: plt.savefig(os.path.join(self.plot_dir, f"{col}_outlier_box.png"))
            plt.close()

            plt.figure()
            sns.violinplot(x=df[col])
            plt.title(f"Violin Plot: {col}")
            if self.save_plots: plt.savefig(os.path.join(self.plot_dir, f"{col}_outlier_violin.png"))
            plt.close()
        return {"numeric_columns": list(num_cols)}

    #############################
    # 6. Feature Importance
    #############################
    def feature_importance(self, df, target_col, task_type="classification"):
        if not target_col or target_col not in df.columns:
            return {}
        metrics.auto_mod()
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode categorical target if classification
        if task_type == "classification" and y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        model = RandomForestClassifier(n_estimators=50, random_state=42) if task_type=="classification" else RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        importances = dict(zip(X.columns, model.feature_importances_))
        return importances

    #############################
    # 7. Data Quality Flags
    #############################
    def data_quality_flags(self, df, target_col=None):
        results = {}
        # Skewness
        results["skewness"] = df.skew(numeric_only=True).to_dict()
        metrics.auto_mod()
        # Target imbalance
        if target_col and target_col in df.columns:
            counts = df[target_col].value_counts(normalize=True)
            results["target_imbalance"] = min(counts.values()) < 0.2

        # Multicollinearity (VIF)
        try:
            num_cols = df.select_dtypes(include=np.number).dropna(axis=1)
            vif_data = pd.DataFrame()
            vif_data["feature"] = num_cols.columns
            vif_data["VIF"] = [variance_inflation_factor(num_cols.values, i)
                               for i in range(num_cols.shape[1])]
            results["VIF"] = vif_data.set_index("feature")["VIF"].to_dict()
        except Exception as e:
            results["VIF_error"] = str(e)
        return results

    #############################
    # 8. HTML/Markdown Report
    #############################
    def generate_report(self, eda_results, report_file="eda_report.html"):
        html = "<html><head><title>EDA Report</title></head><body>"
        html += "<h1>Exploratory Data Analysis Report</h1>"
        metrics.auto_mod()
        for section, result in eda_results.items():
            html += f"<h2>{section.replace('_',' ').title()}</h2>"
            html += f"<pre>{result}</pre>"
        html += "</body></html>"

        with open(report_file, "w") as f:
            f.write(html)
        return report_file

    #############################
    # 9. Run All Steps
    #############################
    def run(self, target_col=None, task_type="classification"):
        self.context.log("Starting EDA Module...")
        metrics.auto_mod()
        try:
            df = getattr(self.context, "transformed_data", None)
            if df is None:
                raise ValueError("No transformed_data found in context.")
            if df.empty:
                raise ValueError("Transformed data is empty.")

            eda_results = {}
            eda_results["data_overview"] = self.data_overview(df)
            eda_results["univariate_analysis"] = self.univariate_analysis(df)
            eda_results["bivariate_analysis"] = self.bivariate_analysis(df, target_col)
            eda_results["target_analysis"] = self.target_analysis(df, target_col)
            eda_results["outlier_analysis"] = self.outlier_analysis(df)
            eda_results["feature_importance"] = self.feature_importance(df, target_col, task_type)
            eda_results["data_quality_flags"] = self.data_quality_flags(df, target_col)
            eda_results["automated_eda_tools"] = self.automated_eda_tools(df, target_col)

            # Save results in context
            self.context.eda_results = eda_results
            self.context.status["eda"] = "completed"
            self.context.log(" EDA completed successfully.")

            # Optional report
            report_path = self.generate_report(eda_results)
            self.context.log(f" EDA report generated: {report_path}")

            # Optional LLM suggestion
            if self.llm_agent:
                metrics.prompt_used()
                suggestion = self.llm_agent.ask(f"EDA summary: {eda_results}. Suggest further insights or checks.")
                self.context.log(f" LLM Suggestion: {suggestion}")

            return True  # ✅ Return True for success

        except Exception as e:
            self.context.status["eda"] = "failed"
            self.context.log(f" EDA failed: {e}")
            return False  # ✅ Return False for failure
     ###############################################
    # 10. External Automated EDA Tools
    ###############################################
    def automated_eda_tools(self, df, target_col=None):
        results = {}
        metrics.auto_mod()
        try:
            from ydata_profiling import ProfileReport
            profile = ProfileReport(df, title="YData Profiling Report", explorative=True)
            ydata_path = os.path.join(self.plot_dir, "ydata_profiling_report.html")
            profile.to_file(ydata_path)
            results["ydata_profiling"] = ydata_path
        except Exception as e:
            results["ydata_profiling_error"] = str(e)

        try:
            from dataprep.eda import create_report
            dp_path = os.path.join(self.plot_dir, "dataprep_report.html")
            report = create_report(df)
            report.save(dp_path)
            results["dataprep_eda"] = dp_path
        except Exception as e:
            results["dataprep_eda_error"] = str(e)

        try:
            from autoviz.AutoViz_Class import AutoViz_Class
            AV = AutoViz_Class()
            av_path = os.path.join(self.plot_dir, "autoviz")
            os.makedirs(av_path, exist_ok=True)

            AV.AutoViz(
                filename="",
                dfte=df,
                depVar=target_col,
                save_plot_dir=av_path,
                verbose=0,
                chart_format="png"   # forces non-GUI backend
            )

            results["autoviz"] = av_path
        except Exception as e:
            results["autoviz_error"] = str(e)

        return results

    def generate_statistical_summary(self, df):
        """Generate statistical summary."""
        pass

    def generate_visualizations(self, df):
        """Generate visualizations."""
        pass

    def log(self, message: str):
        self.context.log(message)