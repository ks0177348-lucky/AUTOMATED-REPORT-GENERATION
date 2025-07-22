
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from datetime import datetime
import os
import numpy as np

class ReportGenerator:
    def __init__(self, output_dir="reports"):  # ✅ Fixed constructor
        """Initialize the report generator with output directory."""
        self.output_dir = output_dir
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def setup_custom_styles(self):
        """Set up custom paragraph styles for the report."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))

        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        ))

        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))

    def load_csv_data(self, file_path):
        try:
            data = pd.read_csv(file_path)
            print(f"Successfully loaded CSV data: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None

    def load_json_data(self, file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            print(f"Successfully loaded JSON data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None

    def analyze_data(self, data):
        if data is None or data.empty:
            return None

        analysis = {
            'summary_stats': data.describe(),
            'null_values': data.isnull().sum(),
            'data_types': data.dtypes,
            'shape': data.shape,
            'columns': list(data.columns)
        }

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['correlations'] = data[numeric_cols].corr()
            analysis['numeric_columns'] = list(numeric_cols)

        return analysis

    def create_visualizations(self, data, analysis):
        if data is None or data.empty:
            return []

        image_files = []
        plt.style.use('seaborn-v0_8')

        numeric_cols = analysis.get('numeric_columns', [])
        if len(numeric_cols) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(data[numeric_cols[0]], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'Distribution of {numeric_cols[0]}')
            plt.xlabel(numeric_cols[0])
            plt.ylabel('Frequency')
            dist_file = os.path.join(self.output_dir, 'distribution_plot.png')
            plt.savefig(dist_file, dpi=300, bbox_inches='tight')
            plt.close()
            image_files.append(dist_file)

        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(analysis['correlations'], annot=True, cmap='coolwarm', fmt=".2f", center=0)
            plt.title('Correlation Matrix')
            corr_file = os.path.join(self.output_dir, 'correlation_heatmap.png')
            plt.savefig(corr_file, dpi=300, bbox_inches='tight')
            plt.close()
            image_files.append(corr_file)

        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            col = categorical_cols[0]
            value_counts = data[col].value_counts().head(10)

            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(value_counts)), value_counts.values, color='lightcoral')
            plt.title(f'Top 10 Values in {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)

            for bar in bars:
                plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                         f'{int(bar.get_height())}', ha='center', va='bottom')

            bar_file = os.path.join(self.output_dir, 'categorical_bar_chart.png')
            plt.tight_layout()
            plt.savefig(bar_file, dpi=300, bbox_inches='tight')
            plt.close()
            image_files.append(bar_file)

        return image_files

    def generate_pdf_report(self, data, analysis, image_files, output_filename):
        doc = SimpleDocTemplate(
            os.path.join(self.output_dir, output_filename),
            pagesize=A4,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=18
        )

        story = []

        title = Paragraph("Data Analysis Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 12))

        report_info = f"""
        <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Data Shape:</b> {analysis['shape'][0]} rows, {analysis['shape'][1]} columns<br/>
        <b>Data Columns:</b> {', '.join(analysis['columns'])}
        """
        story.append(Paragraph(report_info, self.styles['CustomBody']))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Executive Summary", self.styles['CustomHeading']))

        numeric_cols = analysis.get('numeric_columns', [])
        summary_text = f"""
        This report presents a comprehensive analysis of the dataset with {analysis['shape'][0]} records 
        and {analysis['shape'][1]} features. There are {len(numeric_cols)} numeric columns. 
        """

        if len(numeric_cols) > 0:
            avg_vals = analysis['summary_stats'].loc['mean', numeric_cols]
            summary_text += f"Average values range from {avg_vals.min():.2f} to {avg_vals.max():.2f}."

        story.append(Paragraph(summary_text, self.styles['CustomBody']))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Data Quality Assessment", self.styles['CustomHeading']))
        nulls = analysis['null_values']
        total_nulls = nulls.sum()
        null_percent = (total_nulls / (analysis['shape'][0] * analysis['shape'][1])) * 100

        if total_nulls > 0:
            quality_text = f"{total_nulls} missing values ({null_percent:.2f}%). Affected columns: {', '.join(nulls[nulls > 0].index)}."
        else:
            quality_text = "No missing values found in the dataset."

        story.append(Paragraph(quality_text, self.styles['CustomBody']))
        story.append(Spacer(1, 12))

        if len(numeric_cols) > 0:
            story.append(Paragraph("Summary Statistics", self.styles['CustomHeading']))
            summary_stats = analysis['summary_stats']
            table_data = [['Statistic'] + list(summary_stats.columns)]

            for idx in summary_stats.index:
                row = [idx] + [f"{val:.2f}" if pd.notna(val) else "N/A" for val in summary_stats.loc[idx]]
                table_data.append(row)

            table = Table(table_data, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ]))
            story.append(table)
            story.append(Spacer(1, 12))

        story.append(Paragraph("Data Visualizations", self.styles['CustomHeading']))
        for img_file in image_files:
            if os.path.exists(img_file):
                img = Image(img_file, width=6 * inch, height=4 * inch)
                story.append(img)
                story.append(Spacer(1, 12))

        story.append(Paragraph("Key Insights and Recommendations", self.styles['CustomHeading']))
        insights = """
        1. The dataset is mostly complete with very few missing values.<br/>
        2. The numerical analysis shows stable average values and meaningful variance.<br/>
        3. Categorical and correlation visualizations give actionable insights.<br/>
        <br/><b>Recommendations:</b>
        • Maintain data quality with validation checks<br/>
        • Use correlations to inform feature selection<br/>
        • Conduct regular analysis for ongoing insights
        """
        story.append(Paragraph(insights, self.styles['CustomBody']))

        doc.build(story)
        print(f"Report generated successfully: {os.path.join(self.output_dir, output_filename)}")

    def generate_sample_data(self):
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31')
        data = pd.DataFrame({
            'date': dates,
            'sales_amount': np.random.normal(1000, 200, len(dates)),
            'customer_count': np.random.poisson(50, len(dates)),
            'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], len(dates)),
            'discount_percentage': np.random.uniform(0, 30, len(dates)),
            'satisfaction_score': np.clip(np.random.normal(4.2, 0.8, len(dates)), 1, 5)
        })

        data['sales_amount'] *= (1 + data['discount_percentage'] / 100)
        sample_file = os.path.join(self.output_dir, 'sample_data.csv')
        data.to_csv(sample_file, index=False)
        return sample_file

    def run_full_analysis(self, data_file=None, output_filename='analysis_report.pdf'):
        if data_file is None:
            print("No data file provided. Generating sample data...")
            data_file = self.generate_sample_data()

        if data_file.endswith('.csv'):
            data = self.load_csv_data(data_file)
        elif data_file.endswith('.json'):
            data = self.load_json_data(data_file)
        else:
            print("Unsupported file format.")
            return

        if data is None:
            print("Failed to load data.")
            return

        print("Analyzing data...")
        analysis = self.analyze_data(data)

        print("Creating visualizations...")
        images = self.create_visualizations(data, analysis)

        print("Generating PDF report...")
        self.generate_pdf_report(data, analysis, images, output_filename)

        print(f"Analysis complete! Check the '{self.output_dir}' directory.")


# ✅ Correct main entry point
def main():
    generator = ReportGenerator()
    generator.run_full_analysis()


if __name__ == "__main__":  # ✅ Fixed entry point
    main()
