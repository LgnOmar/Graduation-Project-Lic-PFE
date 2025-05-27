"""
Tests for the evaluation module of JibJob recommendation system.
"""

import unittest
import numpy as np
from jibjob_recommender_system.evaluation.evaluation import RecommendationEvaluator

class TestRecommendationEvaluator(unittest.TestCase):
    """Test cases for the RecommendationEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'evaluation': {
                'k_values': [1, 3, 5, 10]
            }
        }
        self.evaluator = RecommendationEvaluator(config=self.config)
        
        # Sample recommendations and ground truth data
        self.recommendations = {
            'user1': ['job1', 'job2', 'job3', 'job4', 'job5'],
            'user2': ['job6', 'job7', 'job8', 'job9', 'job10'],
            'user3': ['job11', 'job12', 'job13', 'job14', 'job15']
        }
        
        self.ground_truth = {
            'user1': ['job1', 'job3', 'job6'],
            'user2': ['job7', 'job9', 'job11'],
            'user3': ['job2', 'job4', 'job15']
        }
        
        # Sample job categories
        self.job_categories = {
            'job1': 'category1',
            'job2': 'category2',
            'job3': 'category1',
            'job4': 'category3',
            'job5': 'category2',
            'job6': 'category1',
            'job7': 'category2',
            'job8': 'category3',
            'job9': 'category1',
            'job10': 'category2',
            'job11': 'category3',
            'job12': 'category1',
            'job13': 'category2',
            'job14': 'category3',
            'job15': 'category1'
        }
        
        # Sample user profiles (interests)
        self.user_profiles = {
            'user1': ['category1', 'category2'],
            'user2': ['category1', 'category3'],
            'user3': ['category2', 'category3']
        }
        
        # Sample item popularity
        self.item_popularity = {
            'job1': 10,
            'job2': 5,
            'job3': 15,
            'job4': 3,
            'job5': 7,
            'job6': 20,
            'job7': 12,
            'job8': 8,
            'job9': 9,
            'job10': 4,
            'job11': 6,
            'job12': 11,
            'job13': 2,
            'job14': 1,
            'job15': 14
        }
        
        # All available jobs
        self.all_items = ['job' + str(i) for i in range(1, 20)]
    
    def test_calculate_hit_rate(self):
        """Test hit rate calculation."""
        hit_rate_at_1 = self.evaluator.calculate_hit_rate(self.recommendations, self.ground_truth, k=1)
        hit_rate_at_3 = self.evaluator.calculate_hit_rate(self.recommendations, self.ground_truth, k=3)
        
        # For k=1, only user1 has a hit (job1)
        self.assertAlmostEqual(hit_rate_at_1, 1/3)
        
        # For k=3, all users should have at least one hit
        self.assertAlmostEqual(hit_rate_at_3, 1.0)
    
    def test_calculate_precision(self):
        """Test precision calculation."""
        precision_at_3 = self.evaluator.calculate_precision(self.recommendations, self.ground_truth, k=3)
        
        # Expected precision for each user at k=3:
        # user1: 2/3 (job1, job3 are in ground truth)
        # user2: 1/3 (job7 is in ground truth)
        # user3: 0/3 (none of first 3 recommendations are in ground truth)
        # Average: (2/3 + 1/3 + 0/3) / 3 = 1/3
        self.assertAlmostEqual(precision_at_3, 1/3)
    
    def test_calculate_recall(self):
        """Test recall calculation."""
        recall_at_5 = self.evaluator.calculate_recall(self.recommendations, self.ground_truth, k=5)
        
        # Expected recall for each user at k=5:
        # user1: 2/3 (job1, job3 are in recommendations, job6 is not)
        # user2: 2/3 (job7, job9 are in recommendations, job11 is not)
        # user3: 1/3 (job15 is in recommendations, job2 and job4 are not)
        # Average: (2/3 + 2/3 + 1/3) / 3 = 5/9
        self.assertAlmostEqual(recall_at_5, 5/9)
    
    def test_calculate_ndcg(self):
        """Test NDCG calculation."""
        ndcg_at_5 = self.evaluator.calculate_ndcg(self.recommendations, self.ground_truth, k=5)
        
        # This is an approximate test since the exact calculation is complex
        # Just ensure it's returning a value between 0 and 1
        self.assertTrue(0 <= ndcg_at_5 <= 1)
    
    def test_calculate_map(self):
        """Test MAP calculation."""
        map_at_5 = self.evaluator.calculate_map(self.recommendations, self.ground_truth, k=5)
        
        # This is an approximate test since the exact calculation is complex
        # Just ensure it's returning a value between 0 and 1
        self.assertTrue(0 <= map_at_5 <= 1)
    
    def test_calculate_diversity(self):
        """Test diversity calculation."""
        diversity = self.evaluator.calculate_diversity(self.recommendations, self.job_categories, k=3)
        
        # Expected diversity for each user at k=3:
        # user1: 2/3 (categories: category1, category2, category1)
        # user2: 3/3 (categories: category1, category2, category3)
        # user3: 3/3 (categories: category3, category1, category2)
        # Average: (2/3 + 3/3 + 3/3) / 3 = 8/9
        self.assertAlmostEqual(diversity, 8/9)
    
    def test_calculate_coverage(self):
        """Test coverage calculation."""
        coverage = self.evaluator.calculate_coverage(self.recommendations, self.all_items, k=5)
        
        # There are 15 unique jobs in recommendations and 19 total jobs
        # Coverage = 15/19
        self.assertAlmostEqual(coverage, 15/19)
    
    def test_calculate_popularity_bias(self):
        """Test popularity bias calculation."""
        bias = self.evaluator.calculate_popularity_bias(self.recommendations, self.item_popularity, k=3)
        
        # Expected popularity for each user at k=3:
        # user1: (10 + 5 + 15) / 3 = 10
        # user2: (20 + 12 + 8) / 3 = 40/3
        # user3: (6 + 11 + 2) / 3 = 19/3
        # Average: (10 + 40/3 + 19/3) / 3 = (30 + 59/3) / 3 = (90 + 59) / 9 = 149/9
        self.assertAlmostEqual(bias, 149/9)
    
    def test_calculate_serendipity(self):
        """Test serendipity calculation."""
        serendipity = self.evaluator.calculate_serendipity(
            self.recommendations, self.user_profiles, self.job_categories, k=3)
        
        # Expected serendipity for each user at k=3:
        # user1: 0/3 (all recommended categories are in user's interests)
        # user2: 1/3 (category2 is not in user2's interests)
        # user3: 1/3 (category1 is not in user3's interests)
        # Average: (0 + 1/3 + 1/3) / 3 = 2/9
        self.assertAlmostEqual(serendipity, 2/9)
    
    def test_evaluate_all_metrics(self):
        """Test the comprehensive evaluation function."""
        results = self.evaluator.evaluate_all_metrics(
            self.recommendations,
            self.ground_truth,
            self.job_categories,
            self.user_profiles,
            self.item_popularity,
            self.all_items
        )
        
        # Check that all expected metrics are present
        expected_metrics = ['hit_rate', 'precision', 'recall', 'ndcg', 'map', 
                           'diversity', 'coverage', 'popularity_bias', 'serendipity']
        for metric in expected_metrics:
            self.assertIn(metric, results)
            
        # Check that metrics are calculated for all k values
        for metric in results:
            for k in self.config['evaluation']['k_values']:
                self.assertIn(k, results[metric])

if __name__ == '__main__':
    unittest.main()
