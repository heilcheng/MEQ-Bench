"""
Public leaderboard generation for MEQ-Bench evaluation results.

This module provides functionality to generate a static HTML leaderboard from
MEQ-Bench evaluation results. The leaderboard displays overall scores for different
models with detailed breakdowns by audience type and complexity level.

The generated leaderboard includes:
- Overall model rankings
- Audience-specific performance breakdowns
- Complexity-level performance analysis
- Interactive charts and visualizations
- Timestamp and benchmark statistics

Usage:
    python -m src.leaderboard --input results/ --output docs/index.html

Author: MEQ-Bench Team
"""

import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('meq_bench.leaderboard')


class LeaderboardGenerator:
    """Generate static HTML leaderboard from evaluation results"""
    
    def __init__(self):
        self.results_data: List[Dict[str, Any]] = []
        self.benchmark_stats: Dict[str, Any] = {}
        
    def load_results(self, results_dir: Path) -> None:
        """Load all result files from a directory
        
        Args:
            results_dir: Directory containing JSON result files
        """
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        result_files = list(results_dir.glob("*.json"))
        if not result_files:
            raise ValueError(f"No JSON result files found in {results_dir}")
        
        logger.info(f"Found {len(result_files)} result files")
        
        for result_file in result_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Validate required fields
                required_fields = ['model_name', 'total_items', 'audience_scores', 'summary']
                if all(field in data for field in required_fields):
                    self.results_data.append(data)
                    logger.debug(f"Loaded results for {data['model_name']}")
                else:
                    logger.warning(f"Invalid result file format: {result_file}")
                    
            except Exception as e:
                logger.error(f"Error loading {result_file}: {e}")
        
        if not self.results_data:
            raise ValueError("No valid result files were loaded")
        
        logger.info(f"Successfully loaded {len(self.results_data)} evaluation results")
    
    def calculate_leaderboard_stats(self) -> Dict[str, Any]:
        """Calculate overall leaderboard statistics
        
        Returns:
            Dictionary containing leaderboard statistics
        """
        if not self.results_data:
            return {}
        
        # Calculate aggregate statistics
        total_models = len(self.results_data)
        total_evaluations = sum(result['total_items'] for result in self.results_data)
        
        # Audience coverage
        all_audiences = set()
        for result in self.results_data:
            all_audiences.update(result['audience_scores'].keys())
        
        # Complexity coverage
        all_complexities = set()
        for result in self.results_data:
            if 'complexity_scores' in result:
                all_complexities.update(result['complexity_scores'].keys())
        
        # Performance ranges
        overall_scores = [result['summary'].get('overall_mean', 0) for result in self.results_data]
        if overall_scores:
            best_score = max(overall_scores)
            worst_score = min(overall_scores)
            avg_score = sum(overall_scores) / len(overall_scores)
        else:
            best_score = worst_score = avg_score = 0
        
        return {
            'total_models': total_models,
            'total_evaluations': total_evaluations,
            'audiences': sorted(list(all_audiences)),
            'complexity_levels': sorted(list(all_complexities)),
            'best_score': best_score,
            'worst_score': worst_score,
            'average_score': avg_score,
            'last_updated': datetime.now().isoformat()
        }
    
    def rank_models(self) -> List[Dict[str, Any]]:
        """Rank models by overall performance
        
        Returns:
            List of model results sorted by overall performance
        """
        ranked_models = sorted(
            self.results_data,
            key=lambda x: x['summary'].get('overall_mean', 0),
            reverse=True
        )
        
        # Add ranking information
        for i, model in enumerate(ranked_models):
            model['rank'] = i + 1
            model['overall_score'] = model['summary'].get('overall_mean', 0)
        
        return ranked_models
    
    def generate_audience_breakdown(self, ranked_models: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate audience-specific performance breakdown
        
        Args:
            ranked_models: List of ranked model results
            
        Returns:
            Dictionary mapping audience to ranked model performance
        """
        audience_breakdown = {}
        
        # Get all audiences
        all_audiences = set()
        for model in ranked_models:
            all_audiences.update(model['audience_scores'].keys())
        
        for audience in sorted(all_audiences):
            audience_models = []
            
            for model in ranked_models:
                if audience in model['audience_scores']:
                    scores = model['audience_scores'][audience]
                    avg_score = sum(scores) / len(scores) if scores else 0
                    
                    audience_models.append({
                        'model_name': model['model_name'],
                        'score': avg_score,
                        'num_items': len(scores)
                    })
            
            # Sort by score for this audience
            audience_models.sort(key=lambda x: x['score'], reverse=True)
            
            # Add rankings
            for i, model in enumerate(audience_models):
                model['rank'] = i + 1
            
            audience_breakdown[audience] = audience_models
        
        return audience_breakdown
    
    def generate_complexity_breakdown(self, ranked_models: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate complexity-specific performance breakdown
        
        Args:
            ranked_models: List of ranked model results
            
        Returns:
            Dictionary mapping complexity level to ranked model performance
        """
        complexity_breakdown = {}
        
        # Get all complexity levels
        all_complexities = set()
        for model in ranked_models:
            if 'complexity_scores' in model:
                all_complexities.update(model['complexity_scores'].keys())
        
        for complexity in sorted(all_complexities):
            complexity_models = []
            
            for model in ranked_models:
                if 'complexity_scores' in model and complexity in model['complexity_scores']:
                    scores = model['complexity_scores'][complexity]
                    avg_score = sum(scores) / len(scores) if scores else 0
                    
                    complexity_models.append({
                        'model_name': model['model_name'],
                        'score': avg_score,
                        'num_items': len(scores)
                    })
            
            # Sort by score for this complexity level
            complexity_models.sort(key=lambda x: x['score'], reverse=True)
            
            # Add rankings
            for i, model in enumerate(complexity_models):
                model['rank'] = i + 1
            
            complexity_breakdown[complexity] = complexity_models
        
        return complexity_breakdown
    
    def generate_html(self, output_path: Path) -> None:
        """Generate static HTML leaderboard
        
        Args:
            output_path: Path where to save the HTML file
        """
        if not self.results_data:
            raise ValueError("No results data loaded")
        
        # Calculate statistics and rankings
        stats = self.calculate_leaderboard_stats()
        ranked_models = self.rank_models()
        audience_breakdown = self.generate_audience_breakdown(ranked_models)
        complexity_breakdown = self.generate_complexity_breakdown(ranked_models)
        
        # Generate HTML content
        html_content = self._generate_html_template(
            stats, ranked_models, audience_breakdown, complexity_breakdown
        )
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated leaderboard HTML: {output_path}")
    
    def _generate_html_template(self, 
                               stats: Dict[str, Any],
                               ranked_models: List[Dict[str, Any]],
                               audience_breakdown: Dict[str, List[Dict[str, Any]]],
                               complexity_breakdown: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate the complete HTML template"""
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MEQ-Bench Leaderboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏆 MEQ-Bench Leaderboard</h1>
            <p class="subtitle">Evaluating Medical Language Models for Audience-Adaptive Explanations</p>
            <div class="stats-bar">
                <div class="stat-item">
                    <span class="stat-value">{stats['total_models']}</span>
                    <span class="stat-label">Models</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">{stats['total_evaluations']:,}</span>
                    <span class="stat-label">Total Evaluations</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">{len(stats['audiences'])}</span>
                    <span class="stat-label">Audiences</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">{stats['best_score']:.3f}</span>
                    <span class="stat-label">Best Score</span>
                </div>
            </div>
            <p class="last-updated">Last updated: {datetime.fromisoformat(stats['last_updated']).strftime('%Y-%m-%d %H:%M UTC')}</p>
        </header>

        <nav class="tabs">
            <button class="tab-button active" onclick="showTab('overall')">Overall Rankings</button>
            <button class="tab-button" onclick="showTab('audience')">By Audience</button>
            <button class="tab-button" onclick="showTab('complexity')">By Complexity</button>
            <button class="tab-button" onclick="showTab('charts')">Analytics</button>
        </nav>

        <div id="overall-tab" class="tab-content active">
            <h2>Overall Model Rankings</h2>
            <div class="table-container">
                {self._generate_overall_rankings_table(ranked_models)}
            </div>
        </div>

        <div id="audience-tab" class="tab-content">
            <h2>Performance by Audience</h2>
            {self._generate_audience_breakdown_section(audience_breakdown)}
        </div>

        <div id="complexity-tab" class="tab-content">
            <h2>Performance by Complexity Level</h2>
            {self._generate_complexity_breakdown_section(complexity_breakdown)}
        </div>

        <div id="charts-tab" class="tab-content">
            <h2>Analytics & Visualizations</h2>
            <div class="charts-container">
                <div class="chart-item">
                    <h3>Model Performance Comparison</h3>
                    <canvas id="performanceChart" width="400" height="200"></canvas>
                </div>
                <div class="chart-item">
                    <h3>Audience Performance Distribution</h3>
                    <canvas id="audienceChart" width="400" height="200"></canvas>
                </div>
            </div>
        </div>

        <footer>
            <div class="footer-content">
                <p><strong>About MEQ-Bench:</strong> A benchmark for evaluating medical language models on audience-adaptive explanation quality.</p>
                <p>Evaluation metrics include readability, terminology appropriateness, safety compliance, information coverage, and overall quality.</p>
                <p>📧 Contact: <a href="mailto:meq-bench@example.com">meq-bench@example.com</a> | 
                   🐙 GitHub: <a href="https://github.com/meq-bench/meq-bench">meq-bench/meq-bench</a></p>
            </div>
        </footer>
    </div>

    <script>
        {self._generate_javascript(ranked_models, audience_breakdown, stats)}
    </script>
</body>
</html>"""
    
    def _get_css_styles(self) -> str:
        """Return CSS styles for the leaderboard"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8fafc;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }

        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 1.5rem;
        }

        .stats-bar {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 1.5rem 0;
            flex-wrap: wrap;
        }

        .stat-item {
            text-align: center;
        }

        .stat-value {
            display: block;
            font-size: 1.8rem;
            font-weight: 700;
            color: #ffd700;
        }

        .stat-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }

        .last-updated {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 1rem;
        }

        .tabs {
            display: flex;
            margin-bottom: 2rem;
            border-bottom: 2px solid #e2e8f0;
            background: white;
            border-radius: 8px 8px 0 0;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .tab-button {
            flex: 1;
            padding: 1rem 1.5rem;
            border: none;
            background: white;
            color: #64748b;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
            border-bottom: 3px solid transparent;
        }

        .tab-button:hover {
            background: #f1f5f9;
            color: #334155;
        }

        .tab-button.active {
            color: #3b82f6;
            border-bottom-color: #3b82f6;
            background: #f8fafc;
        }

        .tab-content {
            display: none;
            background: white;
            padding: 2rem;
            border-radius: 0 0 12px 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }

        .tab-content.active {
            display: block;
        }

        .table-container {
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }

        th, td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }

        th {
            background: #f8fafc;
            font-weight: 600;
            color: #374151;
            position: sticky;
            top: 0;
        }

        tr:hover {
            background: #f8fafc;
        }

        .rank {
            font-weight: 700;
            color: #3b82f6;
        }

        .score {
            font-weight: 600;
            color: #059669;
        }

        .model-name {
            font-weight: 600;
            color: #1f2937;
        }

        .rank-1 {
            background: linear-gradient(135deg, #ffd700, #ffed4e);
            color: #92400e;
        }

        .rank-2 {
            background: linear-gradient(135deg, #c0c0c0, #e5e7eb);
            color: #374151;
        }

        .rank-3 {
            background: linear-gradient(135deg, #cd7f32, #d97706);
            color: white;
        }

        .audience-section, .complexity-section {
            margin-bottom: 2rem;
            background: #f8fafc;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
        }

        .audience-section h3, .complexity-section h3 {
            color: #1f2937;
            margin-bottom: 1rem;
            text-transform: capitalize;
        }

        .charts-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 2rem;
        }

        .chart-item {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .chart-item h3 {
            margin-bottom: 1rem;
            color: #1f2937;
        }

        footer {
            margin-top: 3rem;
            text-align: center;
            padding: 2rem;
            background: #1f2937;
            color: white;
            border-radius: 12px;
        }

        .footer-content p {
            margin-bottom: 0.5rem;
        }

        .footer-content a {
            color: #60a5fa;
            text-decoration: none;
        }

        .footer-content a:hover {
            text-decoration: underline;
        }

        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }

            h1 {
                font-size: 2rem;
            }

            .stats-bar {
                gap: 1rem;
            }

            .tab-button {
                padding: 0.75rem 1rem;
                font-size: 0.9rem;
            }

            .tab-content {
                padding: 1rem;
            }

            th, td {
                padding: 0.75rem 0.5rem;
                font-size: 0.9rem;
            }

            .charts-container {
                grid-template-columns: 1fr;
            }
        }
        """
    
    def _generate_overall_rankings_table(self, ranked_models: List[Dict[str, Any]]) -> str:
        """Generate the overall rankings table HTML"""
        table_html = """
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Overall Score</th>
                    <th>Items Evaluated</th>
                    <th>Physician</th>
                    <th>Nurse</th>
                    <th>Patient</th>
                    <th>Caregiver</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for model in ranked_models:
            rank_class = ""
            if model['rank'] == 1:
                rank_class = "rank-1"
            elif model['rank'] == 2:
                rank_class = "rank-2"
            elif model['rank'] == 3:
                rank_class = "rank-3"
            
            # Calculate audience averages
            audience_scores = {}
            for audience, scores in model['audience_scores'].items():
                avg_score = sum(scores) / len(scores) if scores else 0
                audience_scores[audience] = avg_score
            
            table_html += f"""
                <tr class="{rank_class}">
                    <td class="rank">#{model['rank']}</td>
                    <td class="model-name">{model['model_name']}</td>
                    <td class="score">{model['overall_score']:.3f}</td>
                    <td>{model['total_items']}</td>
                    <td>{audience_scores.get('physician', 0):.3f}</td>
                    <td>{audience_scores.get('nurse', 0):.3f}</td>
                    <td>{audience_scores.get('patient', 0):.3f}</td>
                    <td>{audience_scores.get('caregiver', 0):.3f}</td>
                </tr>
            """
        
        table_html += """
            </tbody>
        </table>
        """
        
        return table_html
    
    def _generate_audience_breakdown_section(self, audience_breakdown: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate the audience breakdown section HTML"""
        html = ""
        
        for audience, models in audience_breakdown.items():
            html += f"""
            <div class="audience-section">
                <h3>{audience.title()} Audience Rankings</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Model</th>
                            <th>Score</th>
                            <th>Items</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for model in models[:10]:  # Show top 10
                rank_class = ""
                if model['rank'] == 1:
                    rank_class = "rank-1"
                elif model['rank'] == 2:
                    rank_class = "rank-2"
                elif model['rank'] == 3:
                    rank_class = "rank-3"
                
                html += f"""
                    <tr class="{rank_class}">
                        <td class="rank">#{model['rank']}</td>
                        <td class="model-name">{model['model_name']}</td>
                        <td class="score">{model['score']:.3f}</td>
                        <td>{model['num_items']}</td>
                    </tr>
                """
            
            html += """
                    </tbody>
                </table>
            </div>
            """
        
        return html
    
    def _generate_complexity_breakdown_section(self, complexity_breakdown: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate the complexity breakdown section HTML"""
        html = ""
        
        for complexity, models in complexity_breakdown.items():
            html += f"""
            <div class="complexity-section">
                <h3>{complexity.title()} Complexity Level Rankings</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Model</th>
                            <th>Score</th>
                            <th>Items</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for model in models[:10]:  # Show top 10
                rank_class = ""
                if model['rank'] == 1:
                    rank_class = "rank-1"
                elif model['rank'] == 2:
                    rank_class = "rank-2"
                elif model['rank'] == 3:
                    rank_class = "rank-3"
                
                html += f"""
                    <tr class="{rank_class}">
                        <td class="rank">#{model['rank']}</td>
                        <td class="model-name">{model['model_name']}</td>
                        <td class="score">{model['score']:.3f}</td>
                        <td>{model['num_items']}</td>
                    </tr>
                """
            
            html += """
                    </tbody>
                </table>
            </div>
            """
        
        return html
    
    def _generate_javascript(self, 
                           ranked_models: List[Dict[str, Any]], 
                           audience_breakdown: Dict[str, List[Dict[str, Any]]],
                           stats: Dict[str, Any]) -> str:
        """Generate JavaScript for interactive features"""
        
        # Prepare data for charts
        model_names = [model['model_name'][:20] for model in ranked_models[:8]]  # Top 8 models
        model_scores = [model['overall_score'] for model in ranked_models[:8]]
        
        audience_labels = list(audience_breakdown.keys())
        audience_data = []
        for audience in audience_labels:
            if audience_breakdown[audience]:
                avg_score = sum(model['score'] for model in audience_breakdown[audience]) / len(audience_breakdown[audience])
                audience_data.append(avg_score)
            else:
                audience_data.append(0)
        
        return f"""
        function showTab(tabName) {{
            // Hide all tab contents
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(tab => tab.classList.remove('active'));
            
            // Remove active class from all buttons
            const tabButtons = document.querySelectorAll('.tab-button');
            tabButtons.forEach(btn => btn.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
            
            // Initialize charts if analytics tab is selected
            if (tabName === 'charts') {{
                setTimeout(initCharts, 100);
            }}
        }}
        
        function initCharts() {{
            // Performance comparison chart
            const performanceCtx = document.getElementById('performanceChart').getContext('2d');
            new Chart(performanceCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(model_names)},
                    datasets: [{{
                        label: 'Overall Score',
                        data: {json.dumps(model_scores)},
                        backgroundColor: 'rgba(59, 130, 246, 0.8)',
                        borderColor: 'rgba(59, 130, 246, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 1
                        }}
                    }}
                }}
            }});
            
            // Audience performance chart
            const audienceCtx = document.getElementById('audienceChart').getContext('2d');
            new Chart(audienceCtx, {{
                type: 'radar',
                data: {{
                    labels: {json.dumps(audience_labels)},
                    datasets: [{{
                        label: 'Average Performance',
                        data: {json.dumps(audience_data)},
                        backgroundColor: 'rgba(16, 185, 129, 0.2)',
                        borderColor: 'rgba(16, 185, 129, 1)',
                        borderWidth: 2,
                        pointBackgroundColor: 'rgba(16, 185, 129, 1)'
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        r: {{
                            beginAtZero: true,
                            max: 1
                        }}
                    }}
                }}
            }});
        }}
        
        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {{
            // Initialize charts if analytics tab is shown by default
            if (document.getElementById('charts-tab').classList.contains('active')) {{
                initCharts();
            }}
        }});
        """


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Generate static HTML leaderboard from MEQ-Bench evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate leaderboard from results directory
    python -m src.leaderboard --input results/ --output docs/index.html
    
    # Generate with custom title
    python -m src.leaderboard --input results/ --output leaderboard.html --title "Custom MEQ-Bench Results"
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Directory containing JSON evaluation result files'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='docs/index.html',
        help='Output path for the HTML leaderboard (default: docs/index.html)'
    )
    
    parser.add_argument(
        '--title',
        type=str,
        default='MEQ-Bench Leaderboard',
        help='Custom title for the leaderboard page'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main():
    """Main function for command-line usage"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize leaderboard generator
        generator = LeaderboardGenerator()
        
        # Load results
        results_dir = Path(args.input)
        generator.load_results(results_dir)
        
        # Generate HTML leaderboard
        output_path = Path(args.output)
        generator.generate_html(output_path)
        
        logger.info(f"✅ Leaderboard generated successfully: {output_path}")
        logger.info(f"📊 Processed {len(generator.results_data)} model results")
        
    except Exception as e:
        logger.error(f"❌ Error generating leaderboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()